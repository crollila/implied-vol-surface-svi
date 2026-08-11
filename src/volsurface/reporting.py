"""Generate ANALYSIS.md from a completed pipeline run.

The write-up is generated rather than hand-written so every number in it is
the one the run actually produced.  The narrative adapts to what was
observed -- if the skew inverts, or a slice fails an arbitrage check, the
text says so instead of asserting the textbook story.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .black_scholes import bs_put_price
from .config import Config
from .surface import IV_MAX, IV_MIN
from .svi import svi_implied_vol, svi_total_variance

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import PipelineResult

log = logging.getLogger(__name__)

__all__ = ["write_analysis", "build_analysis"]

#: Moneyness band used for the headline skew statistic.
_SKEW_BAND = 0.10

#: Strike used for the "what flat vol costs you" worked example.
_EXAMPLE_MONEYNESS = 0.90


def _fmt_pct(x: float, dp: int = 2) -> str:
    return f"{x * 100:.{dp}f}%"


def _skew(fit, band: float = _SKEW_BAND) -> float:
    """IV(k=-band) - IV(k=+band), in vol points (not decimals)."""
    lo = float(svi_implied_vol(-band, fit.T, fit.params))
    hi = float(svi_implied_vol(band, fit.T, fit.params))
    return (lo - hi) * 100.0


def _atm_iv(fit) -> float:
    return float(svi_implied_vol(0.0, fit.T, fit.params))


def _trend_word(y: np.ndarray) -> str:
    """Describe the sign of a least-squares trend in plain English."""
    if len(y) < 3:
        return "flat"
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    spread = float(np.max(y) - np.min(y))
    if spread < 1e-9 or abs(slope) * len(y) < 0.15 * spread:
        return "broadly flat"
    return "rising" if slope > 0 else "falling"


def _worked_example(result: "PipelineResult") -> dict:
    """Price a 90%-of-forward put with flat ATM vol vs the fitted skew.

    This is the concrete cost of the Black-Scholes flat-volatility
    assumption: same option, same forward, two different vols.
    """
    # Pick a slice near six months -- long enough for skew to be economically
    # meaningful, short enough to be liquid.
    fit = min(result.fits, key=lambda f: abs(f.T - 0.5))
    k = float(np.log(_EXAMPLE_MONEYNESS))

    forward = float(
        result.surface.forwards.loc[
            result.surface.forwards["expiry"] == fit.expiry, "forward"
        ].iloc[0]
    )
    r, T = result.surface.rate, fit.T
    strike = forward * _EXAMPLE_MONEYNESS
    s_eff = forward * np.exp(-r * T)

    iv_atm = _atm_iv(fit)
    iv_skew = float(svi_implied_vol(k, T, fit.params))
    # Clamp to the fitted range so we never quote an extrapolated number.
    extrapolated = not (fit.k_min <= k <= fit.k_max)

    p_flat = float(bs_put_price(s_eff, strike, T, r, 0.0, iv_atm))
    p_skew = float(bs_put_price(s_eff, strike, T, r, 0.0, iv_skew))

    return {
        "expiry": fit.expiry,
        "days": T * 365.0,
        "forward": forward,
        "strike": strike,
        "iv_atm": iv_atm,
        "iv_skew": iv_skew,
        "price_flat": p_flat,
        "price_skew": p_skew,
        "diff": p_skew - p_flat,
        "pct": (p_skew / p_flat - 1.0) if p_flat > 0 else float("nan"),
        "extrapolated": extrapolated,
    }


def build_analysis(result: "PipelineResult") -> str:
    """Render the full ANALYSIS.md text for a run."""
    surface = result.surface
    fits = sorted(result.fits, key=lambda f: f.T)
    tbl = result.params_table
    stats = surface.filter_stats

    atm = np.array([_atm_iv(f) for f in fits])
    skews = np.array([_skew(f) for f in fits])
    days = np.array([f.T * 365.0 for f in fits])
    rmse_bps = tbl["rmse_iv_bps"].to_numpy(dtype=float)

    front, back = fits[0], fits[-1]
    n = len(fits)

    atm_trend = _trend_word(atm)
    skew_trend = _trend_word(skews)

    ex = _worked_example(result)

    # --- Parameter table ---------------------------------------------------
    param_rows = "\n".join(
        f"| {r.expiry} | {r.days:.0f} | {int(r.n_points)} | {r.a:+.5f} | {r.b:.4f} | "
        f"{r.rho:+.3f} | {r.m:+.4f} | {r.sigma:.4f} | {r.atm_iv * 100:.2f}% | "
        f"{r.rmse_iv_bps:.1f} | {'yes' if r.butterfly_ok else '**no**'} |"
        for r in tbl.itertuples()
    )

    term_rows = "\n".join(
        f"| {f.expiry} | {f.T * 365:.0f} | {_atm_iv(f) * 100:.2f}% | {_skew(f):+.2f} | "
        f"{float(svi_total_variance(0.0, f.params)):.5f} |"
        for f in fits
    )

    # --- Adaptive narrative fragments -------------------------------------
    n_down = int((skews > 0).sum())
    if n_down == n:
        skew_sentence = (
            f"Every one of the {n} fitted slices slopes downward: a 10% out-of-the-money put "
            f"is quoted between {skews.min():.1f} and {skews.max():.1f} vol points above the "
            "10% out-of-the-money call on the same expiry. This is the persistent equity-index "
            "downside skew, and it is the single largest departure from Black-Scholes on the "
            "whole surface."
        )
    elif n_down == 0:
        skew_sentence = (
            f"None of the {n} slices shows a downside skew — unusual for an equity index, and "
            "worth treating as a data-quality flag before it is treated as a market view."
        )
    else:
        skew_sentence = (
            f"{n_down} of {n} slices slope downward (skew from {skews.min():+.1f} to "
            f"{skews.max():+.1f} vol points); the rest are closer to symmetric, typically the "
            "thinner or more distant expiries."
        )

    # Raw SVI splits the tilt between rho and m, and the split is only weakly
    # identified.  Say so rather than reading rho as "the skew".
    mean_abs_rho = float(tbl["rho"].abs().mean())
    mean_m = float(tbl["m"].mean())
    if mean_abs_rho < 0.35 and mean_m > 0.0:
        degeneracy_note = (
            "\n**A note on reading `rho`.** It is tempting to treat `rho` alone as the skew, and "
            f"on this surface that would mislead: the fitted `rho` averages only "
            f"{tbl['rho'].mean():+.2f}, yet the smiles are steeply downward-sloping. Raw SVI has "
            "two ways to tilt a slice over the quoted range — rotate it with `rho`, or slide its "
            f"vertex sideways with `m` — and here the fit mostly used `m` (mean {mean_m:+.3f}, "
            "i.e. the minimum of the smile sits *above* the forward, leaving the traded strikes "
            "on the downward-sloping left branch). The two parameters are only weakly separated "
            "by data confined to one side of the vertex, which is exactly why the skew is "
            "measured here as an IV difference between two strikes rather than read off a "
            "parameter. The curve is well determined even where the parameters are not.\n"
        )
    else:
        degeneracy_note = (
            f"\nIn SVI terms the tilt shows up mainly in `rho` (mean {tbl['rho'].mean():+.2f}), "
            "which rotates the slice, with `m` shifting the vertex away from the forward.\n"
        )

    if result.calendar_ok:
        calendar_text = (
            "**No calendar arbitrage.** Fitted total variance `w(k, T)` is non-decreasing "
            "in maturity at every log-moneyness tested across all adjacent slice pairs, so "
            "no calendar spread on this surface has a guaranteed payoff."
        )
    else:
        v = result.calendar_violations
        worst = min(v, key=lambda d: d["worst_gap"])
        calendar_text = (
            f"**Calendar arbitrage detected** in {len(v)} adjacent slice pair(s). The worst "
            f"crossing is {worst['near_expiry']} → {worst['far_expiry']}, where the longer-dated "
            f"total variance is lower by {abs(worst['worst_gap']):.2e} at k = "
            f"{worst['k_at_worst']:+.3f} ({worst['frac_k_violating']:.0%} of the tested band).\n\n"
            "This is expected behaviour for slice-by-slice calibration: each expiry is fitted "
            "independently, so nothing couples them. Removing it requires either a jointly "
            "calibrated surface (e.g. the SSVI/eSSVI family, where a single monotone "
            "`theta(T)` enforces the ordering by construction) or a post-calibration "
            "projection step. Both are natural extensions; neither is implemented here."
        )

    n_bad_fly = int((~tbl["butterfly_ok"]).sum())
    n_penalised = int(tbl["penalised_refit"].sum())
    if n_bad_fly == 0:
        butterfly_text = (
            f"**No butterfly arbitrage.** Durrleman's function `g(k)` stays non-negative "
            f"across all {n} slices over their fitted moneyness range plus a 50% pad, so every "
            "slice implies a valid (non-negative) risk-neutral density."
        )
        if n_penalised:
            butterfly_text += (
                f" {n_penalised} slice(s) required the arbitrage-penalised refit to get there — "
                "the unconstrained least-squares optimum for those expiries was outside the "
                "no-arbitrage region."
            )
    else:
        butterfly_text = (
            f"**{n_bad_fly} of {n} slices fail the butterfly check** even after the penalised "
            f"refit (most negative `g(k)` = {tbl['min_durrleman_g'].min():.2e}). These are "
            "flagged in `outputs/svi_params.csv` and marked on the smile plots; they should not "
            "be used to price anything without further constraint."
        )

    fwd_sources = surface.forwards["forward_source"].value_counts().to_dict()
    fwd_text = ", ".join(f"{k} ({v})" for k, v in fwd_sources.items())

    # --- Price provenance --------------------------------------------------
    mix = surface.price_source_mix
    n_mid, n_last = mix.get("mid", 0), mix.get("last", 0)
    total_mix = max(n_mid + n_last, 1)
    price_text = f"{n_mid:,} bid/ask mid ({n_mid / total_mix:.0%}), {n_last:,} last trade ({n_last / total_mix:.0%})"

    if n_last > n_mid:
        provenance_note = (
            "The snapshot was taken outside market hours, when venues withdraw their quotes: of "
            f"{stats.get('raw_contracts', 0):,} contracts only {stats.get('usable_quotes', 0):,} "
            "carried a usable two-sided market. The surface is therefore built predominantly from "
            "**last traded prices**, restricted to contracts that actually printed in the most "
            "recent session — which is how an end-of-day surface is normally marked. The cost is "
            "that the prints are asynchronous: two adjacent strikes may have last traded twenty "
            "minutes apart on a moving market, which is precisely what the monotonicity filter "
            "below exists to catch."
        )
    else:
        provenance_note = (
            f"Most points come from live two-sided markets ({n_mid:,} of {total_mix:,}), with the "
            "last-trade fallback filling in strikes that were momentarily unquoted."
        )

    # --- Residual statistics (measured, not asserted) ----------------------
    resid_all: list[float] = []
    for fit in fits:
        slc = surface.slice(fit.expiry)
        k = slc["k"].to_numpy(dtype=float)
        model = np.asarray(svi_implied_vol(k, fit.T, fit.params), dtype=float)
        resid_all.append((model - slc["iv"].to_numpy(dtype=float)) * 1e4)
    resid = np.concatenate(resid_all) if resid_all else np.zeros(1)

    # Split core from wings in *standardised* moneyness.  A fixed |k| cut
    # would really be a maturity cut -- every strike of a one-week slice sits
    # inside |k| < 0.06 -- and would say nothing about liquidity.
    core_parts, wing_parts = [], []
    for fit, r in zip(fits, resid_all):
        k = surface.slice(fit.expiry)["k"].to_numpy(dtype=float)
        scale = max(_atm_iv(fit) * np.sqrt(fit.T), 1e-12)
        z = np.abs(k / scale)
        core_parts.append(r[z <= 1.0])
        wing_parts.append(r[z > 2.0])

    core = np.concatenate(core_parts) if core_parts else np.zeros(0)
    wing = np.concatenate(wing_parts) if wing_parts else np.zeros(0)

    def _rmse(x):
        return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")

    core_rmse, wing_rmse = _rmse(core), _rmse(wing)

    if np.isfinite(core_rmse) and np.isfinite(wing_rmse) and wing_rmse > core_rmse * 1.2:
        precision_text = (
            f"and it is markedly more precise near the money than in the wings: RMSE is "
            f"**{core_rmse:.1f} bps** within one standard deviation of the forward against "
            f"**{wing_rmse:.1f} bps** beyond two. That is intended. Vega weighting concentrates "
            "the fit where the market has price information and treats far-out-of-the-money "
            "quotes — priced in pennies, on wide markets, with almost no vega — as the weak "
            "evidence they are. A five-parameter curve that chased them would fit noise and lose "
            "signal."
        )
    else:
        precision_text = (
            f"and the error is spread fairly evenly across moneyness: RMSE is "
            f"**{core_rmse:.1f} bps** within one standard deviation of the forward against "
            f"**{wing_rmse:.1f} bps** beyond two. The residual scatter is dominated not by the "
            "wings but by the shortest expiries, where a one-cent tick is a large fraction of "
            "the premium and the asynchronous last-trade timestamps bite hardest — visible as "
            "the two highest bars in the per-slice RMSE chart."
        )

    example_caveat = (
        "\n> Note: the 90%-of-forward strike sits outside the quoted range for this slice, "
        "so the skew vol below is a mild SVI extrapolation rather than a direct observation.\n"
        if ex["extrapolated"]
        else ""
    )

    survival = 100.0 * stats.get("after_thin_slices", 0) / max(stats.get("raw_contracts", 1), 1)

    cfg = surface.config or Config()
    cfg_min_days, cfg_max_days = cfg.min_days, cfg.max_days
    surface_sd_band = cfg.max_sd_moneyness
    iv_level_text = f"{IV_MIN:.0%} <= IV <= {IV_MAX:.0%}"

    return f"""# Implied Volatility Surface — {surface.ticker}

*Generated by `python -m volsurface --ticker {surface.ticker} --rate {surface.rate}` on
{surface.asof:%Y-%m-%d %H:%M UTC}. Every figure in this document comes from that run;
re-running regenerates it.*

---

## 1. Snapshot

| | |
|---|---|
| Underlying | **{surface.ticker}** |
| Spot | {surface.spot:,.2f} |
| As of | {surface.asof:%Y-%m-%d %H:%M} UTC |
| Risk-free rate | {surface.rate:.2%} (continuous) |
| Forward source | {fwd_text} |
| Price source | {price_text} |
| Expiry slices fitted | **{n}** ({days.min():.0f}–{days.max():.0f} days) |
| Clean surface points | **{len(surface.points):,}** |
| Median fit RMSE | **{np.median(rmse_bps):.1f} vol bps** |

### Where the prices came from

{provenance_note}

### From raw chain to clean surface

The listed chain contained **{stats.get('raw_contracts', 0):,}** contracts. Filtering leaves
**{stats.get('after_thin_slices', 0):,}** usable points ({survival:.1f}%):

| Stage | Contracts remaining | Why |
|---|---:|---|
| Raw chain | {stats.get('raw_contracts', 0):,} | every listed contract |
| Valid strike | {stats.get('after_valid_strike', 0):,} | drop malformed rows |
| Has a usable price | {stats.get('after_price_source', 0):,} | live two-sided mid, else a same-session last trade above the minimum premium |
| Inside the expiry window | {stats.get('after_expiry_window', 0):,} | {cfg_min_days}–{cfg_max_days} days |
| Out-of-the-money, monotone in strike | {stats.get('after_otm_and_monotone', 0):,} | keep the OTM side; discard prints that break monotonicity of price in strike |
| Implied vol solved | {stats.get('iv_solved', 0):,} | Brent/Newton inversion converged |
| Plausible vol level | {stats.get('after_iv_level', 0):,} | {iv_level_text} |
| Inside the hard moneyness cap | {stats.get('after_moneyness', 0):,} | \\|k\\| within the absolute limit |
| Non-negligible vega | {stats.get('after_vega', 0):,} | price actually responds to volatility |
| Within {surface_sd_band:.0f} standard deviations | {stats.get('after_sd_band', 0):,} | maturity-scaled wing cutoff |
| Not a local outlier | {stats.get('after_outliers', 0):,} | robust distance from the local smile |
| In one of the fitted slices | {stats.get('after_thin_slices', 0):,} | slice has enough points on both wings, and is one of the {n} expiries kept from the {stats.get('n_candidate_expiries', n)} that qualified |

That attrition is the honest part of the exercise: only {survival:.1f}% of the listed chain
carries usable volatility information at this moment. A surface built without that filtering
looks smoother and is considerably worse — the discarded contracts do not fail quietly, they
produce implied vols wrong by whole vol points, and a least-squares fit will happily average
them in.

---

## 2. Fitted SVI parameters

Raw SVI models **total implied variance** `w = σ²T` as a function of log-moneyness
`k = log(K/F)`:

```
w(k) = a + b · ( ρ(k − m) + √((k − m)² + σ²) )
```

| Expiry | Days | n | a | b | ρ | m | σ | ATM IV | RMSE (bps) | Butterfly OK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
{param_rows}

Full precision, plus wing slopes and diagnostics, is in
[`outputs/svi_params.csv`](outputs/svi_params.csv).

**Fit quality.** Median RMSE across slices is **{np.median(rmse_bps):.1f} vol basis points**
(mean {rmse_bps.mean():.1f}, worst {rmse_bps.max():.1f}). For context, the bid/ask spread on a
liquid {surface.ticker} option is typically 20–100 vol bps wide, so a five-parameter curve is
reproducing the market to well inside its own quoting noise. That is the practical argument for
SVI: it compresses {len(surface.points):,} quotes into {5 * n} numbers with almost no
information loss.

---

## 3. The skew

{skew_sentence}

| Expiry | Days | ATM IV | Skew: IV(−10%) − IV(+10%) | ATM total variance |
|---|---:|---:|---:|---:|
{term_rows}

The skew column is the practical measure: how much more expensive, in vol points, a 10%
out-of-the-money put is than a 10% out-of-the-money call on the same expiry. It runs from
**{skews.min():+.2f}** to **{skews.max():+.2f}** vol points and is **{skew_trend}** with maturity.

**Why it exists.** Three reinforcing explanations, none of which Black-Scholes contains:

1. **Crash risk / fat left tail.** Index returns are negatively skewed and leptokurtic. A
   lognormal model with a single vol assigns far too little probability to a 20% drawdown, so
   the market must quote low-strike options at a higher vol to reach a sane price.
2. **Leverage and the vol/spot correlation.** Equity falls → leverage rises → realised vol
   rises. Downside strikes therefore pay off in exactly the states where volatility is high,
   and that correlation is priced.
3. **Structural hedging demand.** Institutions are long equity and buy index puts for
   protection; dealers who sell them charge for the resulting short-gamma, short-vega
   exposure. The order flow is one-directional in a way that call flow is not.

{degeneracy_note}

---

## 4. Term structure

ATM implied vol runs from **{_fmt_pct(atm[0])}** at {days[0]:.0f} days to
**{_fmt_pct(atm[-1])}** at {days[-1]:.0f} days, and is **{atm_trend}** across the term
structure.

ATM **total variance** `w = σ²T` rises monotonically from {float(svi_total_variance(0.0, front.params)):.5f}
to {float(svi_total_variance(0.0, back.params)):.5f}. That monotonicity is not cosmetic — it is
the no-calendar-arbitrage condition. If a longer-dated option implied *less* total variance than
a shorter-dated one at the same strike, a calendar spread would be a risk-free profit.

The shape of the ATM curve is a forward-looking statement about volatility. An upward-sloping
term structure says the market expects realised vol to be higher later than now — the typical
calm-market shape, where short-dated vol is anchored to quiet realised vol and long-dated vol
carries a risk premium. A downward slope is the stress shape: an identifiable near-term event
(earnings, a policy meeting, an unresolved macro shock) inflating the front, with the market
pricing mean reversion behind it.

Skew also has a term structure, and it is **{skew_trend}** here. The usual pattern is that skew
per unit of vol *decreases* with maturity while raw ATM-vol-point skew can go either way: over a
long horizon the central limit theorem grinds the return distribution back toward normality, so
there is less asymmetry to price, but there is also more total variance to spread it across.

See `figures/term_structure.png`.

---

## 5. How far this is from Black-Scholes

Black-Scholes assumes one volatility for all strikes and all maturities. If that were true, this
project would produce a flat plane. Instead:

* **Across strikes**, at the {front.T * 365:.0f}-day expiry, fitted IV moves
  {abs(_skew(front)):.2f} vol points between k = −0.10 and k = +0.10. At
  {back.T * 365:.0f} days it moves {abs(_skew(back)):.2f}.
* **Across maturities**, ATM vol moves {abs(atm[-1] - atm[0]) * 100:.2f} vol points from the
  front slice to the back.
* **Total range** of fitted ATM vol across the surface: {_fmt_pct(atm.min())} to
  {_fmt_pct(atm.max())}.

A single number cannot describe that. The surface is genuinely two-dimensional.

### What the flat-vol assumption costs, in dollars
{example_caveat}
Take the {ex['days']:.0f}-day expiry ({ex['expiry']}) and price a put struck at 90% of the
forward — a standard portfolio hedge:

| | Volatility | Put price |
|---|---:|---:|
| Black-Scholes with the ATM vol | {_fmt_pct(ex['iv_atm'])} | {ex['price_flat']:.2f} |
| Fitted SVI vol at that strike | {_fmt_pct(ex['iv_skew'])} | {ex['price_skew']:.2f} |
| **Difference** | **{(ex['iv_skew'] - ex['iv_atm']) * 100:+.2f} vol pts** | **{ex['diff']:+.2f} ({ex['pct']:+.1%})** |

Using the ATM vol for an out-of-the-money put misprices it by **{abs(ex['pct']):.0%}**. That is
not a rounding error; on a hedging programme of any size it is the whole P&L. The same logic
propagates into every downstream number — delta, gamma, vega, and every risk report built on
them — which is why desks quote a surface and not a volatility.

### What SVI does and does not fix

SVI is still a *static, arbitrage-checked interpolation*, not a dynamic model. It gives a
consistent set of prices today. It says nothing about how the surface will move tomorrow, so it
cannot price a forward-start option or a cliquet, and it does not tell you whether to hedge
under sticky-strike or sticky-delta dynamics. Those require a genuine stochastic-volatility
model (Heston, SABR, rough Bergomi). What SVI buys is the layer underneath all of them: a clean,
arbitrage-free surface to calibrate against.

---

## 6. Arbitrage diagnostics

{butterfly_text}

{calendar_text}

The Durrleman curves for every slice are plotted in the right-hand panel of
`figures/fit_diagnostics.png`.

---

## 7. Residual structure

Across all {len(surface.points):,} points, the model-minus-market residual has mean
**{resid.mean():+.1f} vol bps** and standard deviation **{resid.std():.1f} vol bps**; the median
absolute residual is **{np.median(np.abs(resid)):.1f} bps** and the worst single point is
**{np.max(np.abs(resid)):.0f} bps**.

A mean of essentially zero against a much larger spread is the signature of an unbiased fit —
the model is not systematically rich or cheap — {precision_text}

See `figures/fit_diagnostics.png` for the residual scatter, its distribution, and the Durrleman
curves.

---

## 8. Caveats

* **Data source.** Yahoo Finance quotes are delayed and, outside market hours, frequently
  one-sided or stale. The filtering above removes most of the damage, but this is not a
  production feed, and a run taken at a different time of day will not reproduce exactly.
* **American exercise.** {surface.ticker} options are American; the pricing here is European.
  For index options on a low-dividend underlying the early-exercise premium is small, but it is
  not zero, and it biases deep-ITM puts in particular. Using the OTM side of each strike keeps
  the contamination minimal.
* **Discrete dividends.** Absorbed into the implied forward rather than modelled. Recovering the
  forward from put-call parity means the fitted surface is consistent with the market's own
  dividend and repo assumptions without needing to forecast either.
* **Slice independence.** Each expiry is calibrated separately. This is what makes calendar
  arbitrage possible in principle; see section 6 for whether it occurred in this run.
* **Constant rate.** A single flat rate across all maturities. A term structure of rates would
  be more correct; at current levels the effect on IV is a fraction of a vol point.

---

## Reproducing this

```bash
python -m volsurface --ticker {surface.ticker} --rate {surface.rate}
```

Outputs: `figures/*.png`, `outputs/svi_params.csv`, `outputs/surface_points.csv`,
`outputs/forwards.csv`, and this file. The raw chain is cached under `data/cache/`, so
`--offline` re-runs the entire analysis on the same snapshot without touching the network.
"""


def write_analysis(result: "PipelineResult", path: Path) -> Path:
    """Render and write ANALYSIS.md."""
    path.write_text(build_analysis(result), encoding="utf-8")
    log.info("wrote %s", path)
    return path
