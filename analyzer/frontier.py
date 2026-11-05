"""Two long-only portfolios over the pushed watchlist, for the weekly email.

The screener decides *which* names to hold; this decides *how much* of each.
Both portfolios are built from the 5-year daily Avanza closes the live path
already fetches for SMA200 (``get_data`` stashes them on the manager), so this
costs no extra API calls and needs no ``--save``.

* **Min-variance (robust)** -- the lowest-volatility mix of the names. Uses
  only the covariance, never an expected return, which is why it is the robust
  one: covariances are estimable from 5 years of weekly data, mean returns are
  not.
* **Max-Sharpe (uses score)** -- the best expected excess return per unit of
  risk, with the screener's ``blend_score`` as the expected return via
  Grinold-Kahn: ``alpha_i = IC * sigma_i * z_i``. Historical mean returns are
  deliberately never used -- they are backward-looking and the panel measured
  no predictive value in them, whereas ``blend_score`` is the validated rank.

Measured on the snapshots before choosing the knobs below (128 names, 260
weeks 2021-07..2026-07, fit on the first half, evaluated on the second, 60
random 25-name draws):

* Avanza closes are not dividend-adjusted, but that inflates the median
  name's weekly vol by 0.13% (Handelsbanken A, the worst, 12%) and moves the
  min-variance weights by a median 2.6% of the book. It only matters for mean
  returns, which this module does not use, so no adjustment is made.
* ``WEIGHT_CAP`` 0.15 cost nothing out of sample (vol 11.5% capped and
  uncapped) and trimmed the worst draw 15.1% -> 14.6%. Its real job is
  max-Sharpe, where the alpha term would otherwise concentrate the book in the
  top-ranked low-vol names -- a single -30% profit warning at 40% weight is
  -12% on the portfolio, a shock the covariance never sees.

Avanza quotes London names in pounds, not pence, so the conversion here does
its own multiply and must not be routed through ``fx.to_sek`` (which divides
GBP by 100 for the Yahoo leg). Nothing here may raise into the live path.
"""

import html
import json
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analyzer.config import SECTOR_CAP

WEIGHT_CAP = 0.15
# Below two years of common weekly history the covariance is too thin even
# shrunk; names that would push the common window under this are dropped.
MIN_WEEKS = 104
# The blend's measured information coefficient (7 fiscal years, positive 7 of
# 7) -- see the LEGEND_LINES in main.py. Scales alpha; it does not change the
# max-Sharpe weights' ordering, only how much of the book the tilt moves.
ASSUMED_IC = 0.118
WEEKS_PER_YEAR = 52


def _close_series(ohlc):
    """Avanza OHLC frame (``date`` column or date index) -> close Series."""
    if not isinstance(ohlc, pd.DataFrame) or ohlc.empty or "close" not in ohlc:
        return None
    df = ohlc.set_index("date") if "date" in ohlc.columns else ohlc
    s = pd.to_numeric(df["close"], errors="coerce")
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[s.index.notna()].dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s if not s.empty else None


def avanza_close_to_sek(close, currency, rates):
    """Restate an Avanza close series in SEK, or None if it cannot be.

    Forward-fill only, and dates before the rate series starts are dropped --
    the same discipline as ``fx.to_sek``, minus its GBP pence division, which
    is correct for Yahoo and 100x wrong for Avanza.
    """
    if currency == "SEK":
        return close
    if rates is None or not currency or currency not in rates.columns:
        return None
    rate = rates[currency].dropna()
    if rate.empty:
        return None
    aligned = rate.reindex(rate.index.union(close.index)).ffill().reindex(close.index)
    out = (close * aligned).dropna()
    return out if not out.empty else None


def weekly_returns(price_hist, currencies, rates, names, extra_closes=None):
    """Common-window weekly SEK returns for ``names``.

    Returns ``(returns, unconverted, dropped)``: ``unconverted`` lists names
    left in listing currency because no rate was available, ``dropped`` names
    with too little history to share a ``MIN_WEEKS`` window with the rest.
    """
    closes, unconverted, dropped = {}, [], []
    for name in names:
        close = _close_series(price_hist.get(name))
        if close is None:
            dropped.append(name)
            continue
        sek = avanza_close_to_sek(close, currencies.get(name), rates)
        if sek is None:
            unconverted.append(name)
            sek = close
        closes[name] = sek
    closes.update(extra_closes or {})
    common, short = common_weekly_returns(closes)
    return common, unconverted, dropped + short


def common_weekly_returns(closes, min_weeks=MIN_WEEKS):
    """``(returns, dropped)``: weekly returns over the longest window all kept
    names share, dropping the shortest histories until it reaches
    ``min_weeks``. Shared with analyzer/evaluation.py's backtest."""
    rets = pd.DataFrame({
        n: s.resample("W-FRI").last().pct_change(fill_method=None)
        for n, s in closes.items()
    }).iloc[1:]
    dropped = []
    while not rets.empty:
        common = rets.dropna()
        if len(common) >= min_weeks:
            return common, dropped
        shortest = rets.count().idxmin()
        dropped.append(shortest)
        rets = rets.drop(columns=shortest)
    return pd.DataFrame(), dropped


def ledoit_wolf(X):
    """Ledoit-Wolf (2004) shrinkage toward a scaled identity.

    Returns ``(covariance, delta)``. Positive definite whenever delta > 0,
    which it always is when observations are fewer than assets.
    """
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0)
    t, n = X.shape
    S = X.T @ X / t
    mu = np.trace(S) / n
    F = mu * np.eye(n)
    d2 = np.sum((S - F) ** 2)
    b2 = sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / t ** 2
    delta = 1.0 if d2 == 0 else float(np.clip(b2 / d2, 0.0, 1.0))
    return delta * F + (1 - delta) * S, delta


def sector_group(sectors):
    """Avanza's ``sectors`` list (or its JSON string, as in a snapshot) ->
    the middle level's name, falling back to the deepest level present.
    None when there is no sector data."""
    if isinstance(sectors, str):
        try:
            sectors = json.loads(sectors)
        except ValueError:
            return None
    if not isinstance(sectors, list):
        return None
    names = [s.get("sectorName") for s in sectors
             if isinstance(s, dict) and s.get("sectorName")]
    if not names:
        return None
    return names[1] if len(names) > 1 else names[0]


def _group_constraints(groups, cap, sector_cap):
    """SLSQP inequality constraints keeping each group's weight under
    ``sector_cap``, raised just enough to stay feasible when there are too few
    groups to hold the whole book. Names without a group are unconstrained."""
    if groups is None or sector_cap is None or sector_cap >= 1.0:
        return [], None
    idx = {}
    for i, g in enumerate(groups):
        if g is not None and not (isinstance(g, float) and np.isnan(g)):
            idx.setdefault(g, []).append(i)
    if not idx:
        return [], None
    singles = len(groups) - sum(len(v) for v in idx.values())

    def capacity(sc):
        return singles * cap + sum(min(sc, len(v) * cap) for v in idx.values())

    sc = sector_cap
    while capacity(sc) < 1.0 - 1e-9 and sc < 1.0:
        sc = min(1.0, sc + 0.01)
    cons = [{"type": "ineq", "fun": (lambda w, ii=np.array(v), c=sc: c - w[ii].sum()),
             "jac": (lambda w, ii=np.array(v): -np.isin(np.arange(len(w)), ii).astype(float))}
            for v in idx.values() if len(v) * cap > sc]
    return cons, sc


def _solve(S, cap, alpha=None, core=None, groups=None, sector_cap=SECTOR_CAP):
    """Long-only, capped satellite weights: min variance, or max Sharpe given
    alpha -- of the *total* portfolio when ``core`` is given.

    ``core = (c, v_c, k, s)``: each stock's covariance with the core, the
    core's variance, and the core's and satellite's shares of the total.
    Total variance is ``k^2 v_c + 2ks w'c + s^2 w'Sw``; without a core it is
    ``w'Sw`` (k=0, s=1), so the no-core case is exactly the original solver.
    The core contributes no alpha: it is the market baseline the score's
    excess is measured against.

    ``groups`` (one industry label per name, None for unknown) caps each
    group's total at ``sector_cap``; without it the solver is unchanged.
    """
    n = S.shape[0]
    cap = max(cap, 1.0 / n)
    group_cons, _ = _group_constraints(groups, cap, sector_cap)
    w0 = np.full(n, 1.0 / n)
    c, v_c, k, sh = core if core is not None else (np.zeros(n), 0.0, 0.0, 1.0)

    def var(w):
        return k * k * v_c + 2 * k * sh * (w @ c) + sh * sh * (w @ S @ w)

    def dvar(w):
        return 2 * k * sh * c + 2 * sh * sh * (S @ w)

    if alpha is None:
        fun, jac = var, dvar
    else:
        def fun(w):
            return -sh * (w @ alpha) / np.sqrt(var(w))

        def jac(w):
            v = var(w)
            return -(sh * alpha * v - sh * (w @ alpha) * dvar(w) / 2) / v ** 1.5
    res = minimize(fun, w0, jac=jac, method="SLSQP",
                   bounds=[(0.0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
                   + group_cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, cap)
    return w / w.sum()


def grinold_alpha(blend, sigma, names, ic=ASSUMED_IC):
    """Annual expected excess return per name from the screener's rank.

    ``blend`` covers the *whole* scored universe, so z is standardised there:
    the pushed names are all above-average and carry positive alpha, which a
    z taken only inside the 25 would hide by forcing half of them negative.
    """
    b = pd.to_numeric(pd.Series(blend), errors="coerce")
    sd = b.std()
    if not np.isfinite(sd) or sd == 0:
        return None
    z = ((b - b.mean()) / sd).reindex(names)
    if z.isna().any():
        return None
    return ic * sigma * z.to_numpy()


def _stats(w, S, alpha, core=None):
    out = {
        "vol": float(np.sqrt(w @ S @ w * WEEKS_PER_YEAR)),
        "exp": float(w @ alpha) if alpha is not None else None,
        "eff_n": float(1.0 / np.sum(w ** 2)),
    }
    if core is not None:
        c, v_c, k, sh = core
        total = k * k * v_c + 2 * k * sh * (w @ c) + sh * sh * (w @ S @ w)
        out["total_vol"] = float(np.sqrt(total * WEEKS_PER_YEAR))
        out["corr_core"] = float((w @ c) / np.sqrt((w @ S @ w) * v_c))
    return out


CORE_KEY = "__core__"


def withholding_drag(div_yields, domiciles, loss_by_country):
    """Annual return lost to withholding tax on an ISK, per name:
    dividend yield x the share of the dividend that cannot be credited.

    Domicile is the ISIN prefix, so an ADR reads as US and an SDB as SE: both
    get no drag. Right or close for the ADRs/SDBs held so far (Shell is UK,
    0%; Autoliv is US, fully creditable), wrong for e.g. a Danish ADR."""
    out = {}
    for name, y in div_yields.items():
        try:
            y = float(y)
        except (TypeError, ValueError):
            y = 0.0
        if not np.isfinite(y) or y < 0:
            y = 0.0
        out[name] = y * loss_by_country.get(domiciles.get(name), 0.0)
    return pd.Series(out, dtype=float)


def build_portfolios(price_hist, currencies, blend, names, rates, cap=WEIGHT_CAP,
                     core_close=None, satellite_share=1.0, tax_drag=None,
                     sectors=None, sector_cap=SECTOR_CAP):
    """Both portfolios for ``names``, or None when there is too little data.

    With ``core_close`` (a SEK price series of the funds the satellite sits
    next to) and ``satellite_share`` < 1, both portfolios optimise the *total*
    -- core plus satellite -- so a stock that moves like the core earns little
    weight in A however calm it is on its own. A core that cannot share the
    stocks' window is dropped with a note and the satellite is optimised alone.
    """
    extra = {CORE_KEY: core_close} if core_close is not None else {}
    rets, unconverted, dropped = weekly_returns(price_hist, currencies, rates,
                                                names, extra_closes=extra)
    core_note = None
    if core_close is not None and CORE_KEY not in rets.columns:
        core_note = "core history too short -- satellite optimised on its own"
        dropped = [d for d in dropped if d != CORE_KEY]
    has_core = CORE_KEY in rets.columns
    kept = [c for c in rets.columns if c != CORE_KEY]
    if len(kept) < 2:
        return None
    S_full, delta = ledoit_wolf(rets[kept + ([CORE_KEY] if has_core else [])].to_numpy())
    n = len(kept)
    S = S_full[:n, :n]
    sigma = np.sqrt(np.diag(S) * WEEKS_PER_YEAR)
    alpha = grinold_alpha(blend, sigma, kept)
    if alpha is not None and tax_drag is not None:
        # After-tax expected return. Only max-Sharpe reads alpha, so this
        # moves B's weights and leaves A's untouched.
        alpha = alpha - tax_drag.reindex(kept).fillna(0.0).to_numpy()
    core = None
    if has_core and satellite_share < 1.0:
        core = (S_full[:n, n], float(S_full[n, n]), 1.0 - satellite_share,
                satellite_share)

    groups = [sectors.get(k) for k in kept] if sectors else None
    w_mv = _solve(S, cap, core=core, groups=groups, sector_cap=sector_cap)
    out = {
        "names": kept,
        "weeks": len(rets),
        "start": rets.index[0].date(),
        "end": rets.index[-1].date(),
        "delta": delta,
        "cap": max(cap, 1.0 / len(kept)),
        "unconverted": unconverted,
        "dropped": dropped,
        "equal_vol": _stats(np.full(len(kept), 1.0 / len(kept)), S, None)["vol"],
        "minvar": {"weights": pd.Series(w_mv, kept), **_stats(w_mv, S, alpha, core)},
        "maxsharpe": None,
        "core": None,
        "core_note": core_note,
    }
    if core is not None:
        out["core"] = {"share": 1.0 - satellite_share,
                       "vol": float(np.sqrt(core[1] * WEEKS_PER_YEAR))}
    if alpha is not None and np.any(alpha > 0):
        w_ms = _solve(S, cap, alpha, core=core, groups=groups,
                      sector_cap=sector_cap)
        out["maxsharpe"] = {"weights": pd.Series(w_ms, kept),
                            **_stats(w_ms, S, alpha, core)}
    return out


TITLE = "PORTFOLIO WEIGHTS (how much of each pushed stock)"
NOT_COMPUTED = "Not computed: too little common price history."
# Below half a percent reads as 0% at the displayed precision; also the bar
# for being put on the min-var / max-sharpe Avanza watchlists.
MIN_WEIGHT = 0.005


def held_names(weights):
    """Names a portfolio actually holds -- the ones shown with a weight."""
    return [n for n, w in weights.items() if w >= MIN_WEIGHT]


def display_name(name):
    """``"Veidekke 52602"`` -> ``"Veidekke"``: the Avanza id is noise to a reader."""
    return re.sub(r"\s+\d+$", "", str(name))


def _pct(x, signed=False):
    if x is None:
        return "-"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def _summary_rows(p):
    """(label, A, B) for the headline stats -- shared by text and HTML."""
    mv, ms = p["minvar"], p["maxsharpe"]
    rows = []
    core = p.get("core")
    if core:
        share = f"{core['share'] * 100:.0f}% core + {(1 - core['share']) * 100:.0f}% these"
        rows += [
            (f"Total vol / yr ({share})", _pct(mv["total_vol"]),
             _pct(ms["total_vol"]) if ms else "-"),
            ("Correlation with core", f"{mv['corr_core']:.2f}",
             f"{ms['corr_core']:.2f}" if ms else "-"),
        ]
    return rows + [
        ("Volatility / yr" + (" (these alone)" if core else ""),
         _pct(mv["vol"]), _pct(ms["vol"]) if ms else "-"),
        ("Exp. excess return / yr", _pct(mv["exp"], True),
         _pct(ms["exp"], True) if ms else "-"),
        ("Effective no. of stocks", f"{mv['eff_n']:.1f}",
         f"{ms['eff_n']:.1f}" if ms else "-"),
    ]


def _weight_rows(p):
    """``(rows, zero_names)``: per-stock (name, A, B) sorted by combined
    weight, with names at ~0% in both portfolios split out."""
    ms = p["maxsharpe"]
    wa = p["minvar"]["weights"]
    wb = ms["weights"] if ms else pd.Series(0.0, index=wa.index)
    order = (wa + wb).sort_values(ascending=False).index
    rows, zero = [], []
    for name in order:
        if wa[name] < MIN_WEIGHT and wb[name] < MIN_WEIGHT:
            zero.append(display_name(name))
            continue
        a = _pct(wa[name]) if wa[name] >= MIN_WEIGHT else "–"
        b = (_pct(wb[name]) if wb[name] >= MIN_WEIGHT else "–") if ms else "-"
        rows.append((display_name(name), a, b))
    return rows, zero


def _footnotes(p, sync_notes=()):
    core = p.get("core")
    notes = list(sync_notes)
    if core:
        notes.append(f"Core = {core.get('label', 'your funds')}, "
                     f"{core['share'] * 100:.0f}% of the total, vol "
                     f"{_pct(core['vol'])} / yr on its own. Weights above are "
                     "shares of the satellite.")
    notes += [
        f"Cap {p['cap'] * 100:.0f}% per stock. {p['weeks']} weeks of Avanza closes "
        f"{p['start']}..{p['end']}, in SEK, Ledoit-Wolf shrinkage {p['delta']:.2f}.",
        f"Exp. return = IC {ASSUMED_IC} x vol x score z-score, i.e. excess vs the "
        "average scored stock, not a forecast of the market.",
    ]
    if not p["maxsharpe"]:
        notes.append("B not computed: no usable score for these stocks.")
    if p["unconverted"]:
        notes.append("[!] No FX rate, left in listing currency: "
                     + ", ".join(map(display_name, p["unconverted"])))
    if p["dropped"]:
        notes.append("[!] Too little price history, left out: "
                     + ", ".join(map(display_name, p["dropped"])))
    return notes


def _explain_pair(p):
    if p.get("core"):
        return ("the stocks that make your WHOLE portfolio (funds + these) as calm "
                "as possible -- favours stocks that move unlike your funds. Ignores "
                "the screener score.",
                "best expected return per unit of WHOLE-portfolio risk, with the "
                "screener score as expected return. Less robust: leans on the score "
                "being right.")
    return ("lowest-risk mix. Uses only how the stocks have moved together -- "
            "ignores the screener score.",
            "best expected return per unit of risk, with the screener score as "
            "expected return. Less robust: leans on the score being right.")


def _explain(p):
    import textwrap
    a, b = _explain_pair(p)
    out = []
    for head, body in (("A = MIN-VARIANCE (robust): ", a), ("B = MAX-SHARPE (uses score): ", b)):
        wrapped = textwrap.wrap(head + body, 66)
        out += [wrapped[0]] + [f"    {w}" for w in wrapped[1:]]
    return out


def format_portfolios(p, sync_notes=()):
    """Plain-text block for the terminal and the email's text part."""
    bar = "=" * 50
    lines = [bar, f"  {TITLE}", bar]
    if p is None:
        return lines + [f"  {NOT_COMPUTED}"]

    lines += [f"  {line}" for line in _explain(p)] + [""]
    rows, zero = _weight_rows(p)
    w = max([28] + [len(r[0]) for r in rows])

    def _row(label, a, b):
        lines.append(f"  {label:<{w}}{a:>11}{b:>14}")

    _row("", "A min-var", "B max-Sharpe")
    for r in _summary_rows(p):
        _row(*r)
    lines.append(f"  (equal weight: volatility {_pct(p['equal_vol'])} / yr)")
    lines.append("")
    for r in rows:
        _row(*r)
    if zero:
        lines.append(f"  0% in both ({len(zero)}): {', '.join(zero)}")
    lines.append("")
    lines += [f"  {n}" for n in _footnotes(p, sync_notes)]
    return lines


_TH = "padding:6px 10px;text-align:right;border-bottom:2px solid #444;"
_TD = "padding:4px 10px;text-align:right;font-variant-numeric:tabular-nums;"
_TDL = "padding:4px 10px;text-align:left;"


def format_portfolios_html(p, sync_notes=()):
    """The same content as an HTML table, for the email's HTML part.

    Inline styles only -- Gmail strips <style> blocks.
    """
    e = html.escape
    out = [f'<h3 style="margin:24px 0 6px;font-family:sans-serif">{e(TITLE)}</h3>']
    if p is None:
        return "\n".join(out + [f'<p style="font-family:sans-serif">{e(NOT_COMPUTED)}</p>'])

    a, b = _explain_pair(p)
    out.append(
        '<p style="font-family:sans-serif;font-size:13px;margin:0 0 10px;line-height:1.5">'
        f"<b>A = Min-variance (robust)</b>: {e(a)}<br>"
        f"<b>B = Max-Sharpe (uses score)</b>: {e(b)}</p>")
    t = ['<table style="border-collapse:collapse;font-family:sans-serif;font-size:13px">',
         "<tr>"
         f'<th style="{_TH}text-align:left"></th>'
         f'<th style="{_TH}">A &nbsp;Min-variance<br><span style="font-weight:normal;'
         'color:#666">robust</span></th>'
         f'<th style="{_TH}">B &nbsp;Max-Sharpe<br><span style="font-weight:normal;'
         'color:#666">uses score</span></th></tr>']
    for label, a, b in _summary_rows(p):
        t.append(f'<tr style="background:#f0f0f0"><td style="{_TDL}"><b>{e(label)}</b></td>'
                 f'<td style="{_TD}"><b>{e(a)}</b></td><td style="{_TD}"><b>{e(b)}</b></td></tr>')
    t.append(f'<tr style="background:#f0f0f0"><td style="{_TDL};color:#666" colspan="3">'
             f"Equal weight: volatility {e(_pct(p['equal_vol']))} / yr</td></tr>")
    rows, zero = _weight_rows(p)
    for i, (name, a, b) in enumerate(rows):
        bg = "#fafafa" if i % 2 else "#ffffff"
        top = "border-top:2px solid #444;" if i == 0 else ""
        t.append(f'<tr style="background:{bg}"><td style="{_TDL}{top}">{e(name)}</td>'
                 f'<td style="{_TD}{top}">{e(a)}</td><td style="{_TD}{top}">{e(b)}</td></tr>')
    t.append("</table>")
    out += t
    small = 'style="font-family:sans-serif;font-size:12px;color:#666;margin:6px 0"'
    if zero:
        out.append(f"<p {small}>0% in both ({len(zero)}): {e(', '.join(zero))}</p>")
    out.append(f"<p {small}>" + "<br>".join(e(n) for n in _footnotes(p, sync_notes)) + "</p>")
    return "\n".join(out)
