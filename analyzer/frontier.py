"""Two long-only portfolios over the pushed watchlist, for the weekly email.

The screener decides *which* names to hold; this decides *how much* of each.
Both portfolios are built from the 5-year daily Avanza closes the live path
already fetches for SMA200 (``get_data`` stashes them on the manager), so this
costs no extra API calls and needs no ``--save``.

* **Min-variance (robust)** -- the lowest-volatility mix of the names. Uses
  only the covariance, never an expected return, which is why it is the robust
  one: covariances are estimable from 5 years of weekly data, mean returns are
  not.
* **Max-Sharpe (uses rank)** -- the best expected excess return per unit of
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

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


def weekly_returns(price_hist, currencies, rates, names):
    """Common-window weekly SEK returns for ``names``.

    Returns ``(returns, unconverted, dropped)``: ``unconverted`` lists names
    left in listing currency because no rate was available, ``dropped`` names
    with too little history to share a ``MIN_WEEKS`` window with the rest.
    """
    cols, unconverted, dropped = {}, [], []
    for name in names:
        close = _close_series(price_hist.get(name))
        if close is None:
            dropped.append(name)
            continue
        sek = avanza_close_to_sek(close, currencies.get(name), rates)
        if sek is None:
            unconverted.append(name)
            sek = close
        cols[name] = sek.resample("W-FRI").last().pct_change(fill_method=None)

    rets = pd.DataFrame(cols).iloc[1:]
    # Shortest histories go first until the common window is long enough.
    while not rets.empty:
        common = rets.dropna()
        if len(common) >= MIN_WEEKS:
            return common, unconverted, dropped
        shortest = rets.count().idxmin()
        dropped.append(shortest)
        rets = rets.drop(columns=shortest)
    return pd.DataFrame(), unconverted, dropped


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


def _solve(S, cap, alpha=None):
    """Long-only, capped weights: min variance, or max Sharpe given alpha."""
    n = S.shape[0]
    cap = max(cap, 1.0 / n)
    w0 = np.full(n, 1.0 / n)
    if alpha is None:
        fun = lambda w: w @ S @ w
        jac = lambda w: 2 * S @ w
    else:
        def fun(w):
            return -(w @ alpha) / np.sqrt(w @ S @ w)

        def jac(w):
            v = w @ S @ w
            return -(alpha * v - (w @ alpha) * (S @ w)) / v ** 1.5
    res = minimize(fun, w0, jac=jac, method="SLSQP",
                   bounds=[(0.0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
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


def _stats(w, S, alpha):
    return {
        "vol": float(np.sqrt(w @ S @ w * WEEKS_PER_YEAR)),
        "exp": float(w @ alpha) if alpha is not None else None,
        "eff_n": float(1.0 / np.sum(w ** 2)),
    }


def build_portfolios(price_hist, currencies, blend, names, rates, cap=WEIGHT_CAP):
    """Both portfolios for ``names``, or None when there is too little data."""
    rets, unconverted, dropped = weekly_returns(price_hist, currencies, rates, names)
    if rets.shape[1] < 2:
        return None
    kept = list(rets.columns)
    S, delta = ledoit_wolf(rets.to_numpy())
    sigma = np.sqrt(np.diag(S) * WEEKS_PER_YEAR)
    alpha = grinold_alpha(blend, sigma, kept)

    w_mv = _solve(S, cap)
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
        "minvar": {"weights": pd.Series(w_mv, kept), **_stats(w_mv, S, alpha)},
        "maxsharpe": None,
    }
    if alpha is not None and np.any(alpha > 0):
        w_ms = _solve(S, cap, alpha)
        out["maxsharpe"] = {"weights": pd.Series(w_ms, kept), **_stats(w_ms, S, alpha)}
    return out


def format_portfolios(p):
    """Plain-text block for the email and terminal; names each portfolio."""
    bar = "=" * 50
    lines = [bar, "  PORTFOLIO WEIGHTS (how much of each pushed stock)", bar]
    if p is None:
        return lines + ["  Not computed: too little common price history."]

    ms = p["maxsharpe"]
    lines += [
        "  A = MIN-VARIANCE (robust): lowest-risk mix. Uses only how the",
        "      stocks have moved together -- ignores the screener rank.",
        "  B = MAX-SHARPE (uses rank): best expected return per unit of",
        "      risk, with the screener rank as expected return. Less robust:",
        "      leans on the rank being right.",
        "",
        f"  {'':<28}{'A min-var':>11}{'B max-Sharpe':>14}",
    ]

    def _row(label, a, b):
        lines.append(f"  {label:<28}{a:>11}{b:>14}")

    def _pct(x, signed=False):
        if x is None:
            return "-"
        return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"

    mv = p["minvar"]
    _row("Volatility/yr", _pct(mv["vol"]), _pct(ms["vol"]) if ms else "-")
    _row("Exp. excess return/yr", _pct(mv["exp"], True),
         _pct(ms["exp"], True) if ms else "-")
    _row("Effective no. of stocks", f"{mv['eff_n']:.1f}",
         f"{ms['eff_n']:.1f}" if ms else "-")
    lines.append(f"  (equal weight: volatility {_pct(p['equal_vol'])}/yr)")
    lines.append("")

    wa = mv["weights"]
    wb = ms["weights"] if ms else pd.Series(0.0, index=wa.index)
    order = (wa + wb).sort_values(ascending=False).index
    zero = []
    for name in order:
        if wa[name] < 0.005 and wb[name] < 0.005:
            zero.append(name)
            continue
        _row(str(name)[:28], _pct(wa[name]), _pct(wb[name]) if ms else "-")
    if zero:
        lines.append(f"  0% in both ({len(zero)}): {', '.join(map(str, zero))}")

    lines.append("")
    lines.append(f"  Cap {p['cap'] * 100:.0f}% per stock. {p['weeks']} weeks of Avanza "
                 f"closes {p['start']}..{p['end']},")
    lines.append(f"  in SEK, Ledoit-Wolf shrinkage {p['delta']:.2f}. Exp. return = "
                 f"IC {ASSUMED_IC} x vol x rank z-score,")
    lines.append("  i.e. excess vs the average scored stock, not a forecast of the market.")
    if not ms:
        lines.append("  B not computed: no usable rank for these stocks.")
    if p["unconverted"]:
        lines.append(f"  [!] No FX rate, left in listing currency: "
                     f"{', '.join(map(str, p['unconverted']))}")
    if p["dropped"]:
        lines.append(f"  [!] Too little price history, left out: "
                     f"{', '.join(map(str, p['dropped']))}")
    return lines
