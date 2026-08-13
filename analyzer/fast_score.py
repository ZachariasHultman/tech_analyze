"""Vectorised re-scoring for the optimizer's inner loop.

The optimizer evaluates one candidate by rebuilding the whole scoring chain --
``SummaryManager`` -> ``process_historical`` -> ``calculate_score`` -- which
re-derives ratios, re-winsorizes, re-ranks and applies row-wise. None of that
depends on the weights. Measured: 132 ms per evaluation, 1278 evaluations per
``optimize_panel_combo`` pass, 168 s total.

Both scoring paths are exactly linear in weight:

* ``SummaryManager._assign_points_rank`` returns ``weight * g(rank)``
* ``SummaryManager._assign_points`` returns ``weight * f(value, thresholds)``
  (including the composite branch, which averages two such terms)

so for a fixed threshold configuration::

    points(w) = G @ w + bonus(w)

where ``G`` is a (companies x metrics) matrix obtained from a *single*
unit-weight scoring pass. Only ``bonus`` needs re-deriving per candidate, and
it depends on ``w`` solely through which metrics are non-zero.

Verified exact against the production path to 3.55e-15 across random weight
vectors (see tests/test_fast_score.py) -- this is a speed refactor with no
behavioural change, which is precisely why the test asserts equality rather
than closeness of downstream statistics.
"""

import numpy as np
import pandas as pd

from analyzer.metrics import HIGHEST_WEIGHT_METRICS

# Bounded so a long coordinate-descent run can't grow this without limit.
# Entries are single columns (~127 floats each, ~1 KB), and a full run touches
# roughly years x metrics x distinct-thresholds of them, so the ceiling is set
# well above that: 20k entries is ~20 MB, cheap against a 168 s -> ~1 s win.
_CACHE_MAX = 20000
_MATRIX_CACHE: dict = {}

# (group_key, metric) pairs whose score column provably does not depend on
# thresholds in that cross-section -- see _probe_threshold_independence.
_THRESHOLD_INDEPENDENT: set = set()


def build_score_matrix(df_slice, metrics, thresholds=None):
    """One unit-weight scoring pass -> ``(G, hw_cols)``.

    ``G`` is a DataFrame indexed by company with one column per metric holding
    that metric's score at weight 1.0. ``hw_cols`` are the highest-weight
    metrics that actually produced a score column, mirroring how
    ``calculate_score`` picks its bonus/malus set (it only considers metrics
    present in the frame, so a missing one shrinks the set rather than
    disabling the bonus).
    """
    # Imported here: correlation.py imports this module, so a module-level
    # import would be circular.
    from analyzer.correlation import _score_snapshot

    scored = _score_snapshot(
        df_slice,
        metrics_to_score=list(metrics),
        thresholds=thresholds,
        weight_overrides={m: 1.0 for m in metrics},
    )
    if scored is None or scored.empty:
        return None, []

    cols = {}
    for m in metrics:
        sc = m + "_score"
        cols[m] = (
            pd.to_numeric(scored[sc], errors="coerce")
            if sc in scored.columns
            else pd.Series(np.nan, index=scored.index)
        )
    G = pd.DataFrame(cols, index=scored.index).fillna(0.0)
    hw_cols = [
        m for m in HIGHEST_WEIGHT_METRICS
        if m in G.columns and (m + "_score") in scored.columns
    ]
    return G, hw_cols


def points_from_matrix(G, hw_cols, weights):
    """``G @ w`` plus the symmetric bonus/malus, as a company-indexed Series.

    The bonus mirrors ``calculate_score``: +1 when every highest-weight metric
    scores strictly positive, -1 when every one scores strictly negative. A
    metric carrying weight 0 scores exactly 0, so it is neither -- which
    correctly suppresses the bonus entirely, same as production.
    """
    metrics = list(G.columns)
    w = np.array([float(weights.get(m, 0.0)) for m in metrics])
    pts = G.to_numpy() @ w

    if hw_cols:
        hw_w = np.array([float(weights.get(m, 0.0)) for m in hw_cols])
        signed = G[hw_cols].to_numpy() * hw_w
        pts = pts + (signed > 0).all(axis=1).astype(int) \
                  - (signed < 0).all(axis=1).astype(int)
    return pd.Series(pts, index=G.index)


def _metric_threshold_key(thresholds, metric):
    """Cache key for one metric's own threshold configuration.

    A metric's score column depends only on *its own* thresholds -- scoring is
    per-metric, and the cross-sectional rank path ignores thresholds entirely.
    Verified empirically: scoring a single metric in isolation reproduces its
    column from a full pass to 0.0 across every metric in the real panel. So
    caching per column (rather than per whole matrix) means a threshold sweep
    on one metric rebuilds one column instead of all of them -- which matters
    because the threshold sweep, not the weight sweep, dominates coordinate
    descent.
    """
    if not thresholds:
        return None
    cfg = thresholds.get(metric)
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return (cfg.get("nok"), cfg.get("ok"))
    return repr(cfg)


def _shifted(thresholds, metrics, direction):
    """Copy of ``thresholds`` with ``metrics``' bands moved well off the data.

    Both endpoints shift together, so the nok/ok ordering (and therefore
    ``_normalize_pair``'s direction handling) is preserved -- a swap would be
    silently undone by that normalisation and make a useless probe.
    """
    out = dict(thresholds or {})
    for m in metrics:
        cfg = out.get(m)
        if not isinstance(cfg, dict):
            continue
        try:
            nok, ok = float(cfg["nok"]), float(cfg["ok"])
        except (KeyError, TypeError, ValueError):
            continue
        delta = direction * max(1.0, abs(ok - nok)) * 3.0
        out[m] = {"nok": nok + delta, "ok": ok + delta}
    return out


def _probe_threshold_independence(df_slice, metrics, thresholds, group_key,
                                  baseline, score_fn):
    """Record which metrics' columns are provably threshold-insensitive here.

    A metric routed through ``calculate_score``'s cross-sectional rank path is
    scored by ``_assign_points_rank``, which never reads thresholds -- so its
    column is identical for every threshold configuration. On the real panel
    that is 12 of 15 metrics (only the near-empty ones fall back to absolute
    thresholds), and coordinate descent spends roughly half its evaluations
    sweeping thresholds. Caching those columns per-threshold means rebuilding
    them for no reason.

    Rather than re-deriving the routing rule here (duplicated logic that could
    drift from calculate_score), this measures it: re-score with the band
    shifted far above and far below and keep only the metrics whose column is
    unchanged both times. Two-sided so a metric that merely happens to be
    insensitive to one direction of shift is not mistaken for a rank-path one.
    """
    if not thresholds:
        return
    candidates = [m for m in metrics if isinstance(thresholds.get(m), dict)]
    if not candidates:
        return
    unchanged = set(candidates)
    for direction in (+1.0, -1.0):
        if not unchanged:
            break
        probe = score_fn(df_slice, sorted(unchanged),
                         _shifted(thresholds, unchanged, direction))
        if probe is None or probe.empty:
            return
        for m in list(unchanged):
            sc = m + "_score"
            if sc not in probe.columns:
                base = baseline.get(m)
                if base is None or (base[0] == 0.0).all():
                    continue
                unchanged.discard(m)
                continue
            col = pd.to_numeric(probe[sc], errors="coerce").fillna(0.0)
            base = baseline.get(m)
            if base is None:
                unchanged.discard(m)
                continue
            aligned = col.reindex(base[0].index).fillna(0.0)
            if not np.allclose(aligned.to_numpy(), base[0].to_numpy(),
                               rtol=0, atol=0):
                unchanged.discard(m)
    for m in unchanged:
        _THRESHOLD_INDEPENDENT.add((group_key, m))


def cached_score_matrix(df_slice, metrics, thresholds=None, group_key=None):
    """Assemble ``(G, hw_cols)`` from a per-column memo, scoring only misses.

    ``group_key`` identifies the cross-section; the panel optimizers pass the
    fiscal year. Falls back to a cheap index fingerprint when not supplied.
    """
    from analyzer.correlation import _score_snapshot

    metrics = list(metrics)
    # A content fingerprint is always folded in, even when the caller supplies
    # a group_key. Otherwise a caller that passed a *subset* of a fiscal year
    # under that year's key would silently receive the full year's matrix --
    # the kind of quiet mismatch this whole change set exists to eliminate.
    # Hashing ~127 company labels costs microseconds against a ~30 ms rebuild.
    fingerprint = (len(df_slice), hash(tuple(df_slice.get("company", df_slice.index))))
    group_key = (group_key, fingerprint)

    def _key(m):
        # A metric proven threshold-insensitive in this cross-section drops
        # the threshold from its key, so a threshold sweep is a cache hit.
        if (group_key, m) in _THRESHOLD_INDEPENDENT:
            return (group_key, m, None)
        return (group_key, m, _metric_threshold_key(thresholds, m))

    def _score(frame, subset, thr):
        return _score_snapshot(
            frame,
            metrics_to_score=list(subset),
            thresholds=thr,
            weight_overrides={m: 1.0 for m in subset},
        )

    # Evict *before* resolving hits. Clearing afterwards would drop entries
    # this very call already counted as hits, leaving the assembly loop with a
    # KeyError.
    if len(_MATRIX_CACHE) + len(metrics) > _CACHE_MAX:
        _MATRIX_CACHE.clear()
        _THRESHOLD_INDEPENDENT.clear()

    keys = {m: _key(m) for m in metrics}
    missing = [m for m in metrics if keys[m] not in _MATRIX_CACHE]

    if missing:
        scored = _score(df_slice, missing, thresholds)
        if scored is None or scored.empty:
            return None, []
        built = {}
        for m in missing:
            sc = m + "_score"
            present = sc in scored.columns
            col = (
                pd.to_numeric(scored[sc], errors="coerce").fillna(0.0)
                if present
                else pd.Series(0.0, index=scored.index)
            )
            # `present` is tracked separately from an all-zero column: a
            # highest-weight metric that produced no column is excluded from
            # the bonus set, matching calculate_score, whereas one that scored
            # zero everywhere still counts (and correctly suppresses the bonus).
            built[m] = (col, present)

        # Probe only metrics never seen before for this cross-section, so the
        # extra passes are paid once per (year, metric), not per sweep.
        unprobed = [m for m in missing if (group_key, m) not in _THRESHOLD_INDEPENDENT]
        if unprobed:
            _probe_threshold_independence(
                df_slice, unprobed, thresholds, group_key, built, _score
            )

        for m in missing:
            _MATRIX_CACHE[_key(m)] = built[m]
        keys = {m: _key(m) for m in metrics}

    cols, hw_cols = {}, []
    for m in metrics:
        col, present = _MATRIX_CACHE[keys[m]]
        cols[m] = col
        if present and m in HIGHEST_WEIGHT_METRICS:
            hw_cols.append(m)
    G = pd.DataFrame(cols)
    if G.empty:
        return None, []
    # Keep hw order aligned with HIGHEST_WEIGHT_METRICS for stable comparisons.
    hw_cols = [m for m in HIGHEST_WEIGHT_METRICS if m in hw_cols]
    return G, hw_cols


def clear_cache():
    """Drop every memoised column. Call between runs over different data."""
    _MATRIX_CACHE.clear()
    _THRESHOLD_INDEPENDENT.clear()
