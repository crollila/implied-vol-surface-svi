"""Turn a raw options chain into a clean implied-volatility surface table.

This module is where most of the real-world care lives.  A listed chain
contains thousands of contracts that are untradeable, stale, or outright
inconsistent; feeding them straight to a calibrator produces a smooth fit to
garbage.  The steps are:

1. **Price source selection.** Prefer the bid/ask mid of a live two-sided
   quote.  Outside market hours most venues withdraw quotes entirely, so
   fall back to the last traded price *provided the contract actually traded
   in the most recent session*.  Which source was used is recorded per row.
2. **Quote hygiene.** Drop crossed markets, absurd relative spreads, sub-tick
   premiums, and contracts that have not traded recently.
3. **Forward estimation.** Recover each expiry's forward from put-call
   parity, sanity-checked against cost-of-carry.  The listed market prices
   the forward directly, so using it removes any dependence on a dividend
   forecast and re-centres the smile correctly.
4. **Out-of-the-money selection.** Use puts below the forward and calls
   above.  In-the-money options are mostly intrinsic value: a one-tick quote
   error on a deep ITM call moves its implied vol by whole vol points, so
   including them injects far more noise than information.
5. **Static-arbitrage filtering.** OTM put prices must rise with strike and
   OTM call prices must fall; asynchronous last trades routinely violate
   this.  Keep the largest mutually consistent subset.
6. **Inversion** to Black-Scholes implied vol, then post-filtering on vol
   level, moneyness, vega, and robust outlier distance from the local smile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .black_scholes import bs_vega
from .config import Config
from .data import ChainSnapshot
from .implied_vol import implied_vol_vectorised

log = logging.getLogger(__name__)

__all__ = ["SurfaceData", "build_surface", "year_fraction", "longest_monotone_subset"]

#: US equity options expire at the 16:00 ET close, i.e. 20:00 UTC during EDT.
_EXPIRY_HOUR_UTC = 20

#: ACT/365 day count, the market convention for equity vol.
_DAYS_PER_YEAR = 365.0

#: Implied vols outside this band are numerical artefacts, not market levels.
IV_MIN, IV_MAX = 0.01, 3.0

#: Below this vega (per 1.0 vol, in price units) a quote's implied vol is
#: dominated by the tick size rather than by information.
_MIN_VEGA = 1e-3

#: A parity forward further than this (in log terms) from the cost-of-carry
#: forward is treated as a bad measurement rather than a market signal.
_MAX_FORWARD_DEVIATION = 0.03

#: Robust outlier cutoff, in MADs from a local quadratic smile fit.
_OUTLIER_MADS = 4.0


@dataclass
class SurfaceData:
    """Cleaned surface points plus the metadata needed to reproduce them."""

    points: pd.DataFrame
    """One row per usable quote.  Key columns: ``expiry``, ``T``, ``strike``,
    ``forward``, ``k`` (log-moneyness vs forward), ``iv``, ``total_var``,
    ``vega``, ``option_type``, ``price``, ``price_source``."""

    forwards: pd.DataFrame
    """One row per expiry: ``expiry``, ``T``, ``days``, ``forward``,
    ``forward_source``, ``n_points``."""

    spot: float
    asof: datetime
    ticker: str
    rate: float
    dividend_yield: float
    filter_stats: dict[str, int] = field(default_factory=dict)
    price_source_mix: dict[str, int] = field(default_factory=dict)

    config: Config | None = None
    """The settings this surface was built with, kept so the write-up can
    quote the actual thresholds rather than restating defaults."""

    @property
    def expiries(self) -> list[str]:
        return self.forwards["expiry"].tolist()

    def slice(self, expiry: str) -> pd.DataFrame:
        """Points for one expiry, sorted by log-moneyness."""
        sub = self.points[self.points["expiry"] == expiry]
        return sub.sort_values("k").reset_index(drop=True)


def year_fraction(expiry: str, asof: datetime) -> float:
    """ACT/365 year fraction from ``asof`` to the 16:00 ET expiry close."""
    exp_dt = datetime.fromisoformat(expiry).replace(
        hour=_EXPIRY_HOUR_UTC, tzinfo=timezone.utc
    )
    if asof.tzinfo is None:
        asof = asof.replace(tzinfo=timezone.utc)
    return (exp_dt - asof).total_seconds() / (_DAYS_PER_YEAR * 86400.0)


def longest_monotone_subset(values: np.ndarray, increasing: bool) -> np.ndarray:
    """Indices of the longest non-strictly-monotone subsequence of ``values``.

    Used to enforce the static-arbitrage requirement that OTM put prices rise
    with strike and OTM call prices fall.  Rather than sweeping once and
    dropping every point that breaks monotonicity (which discards a long run
    of good quotes after a single bad one), this keeps the *largest mutually
    consistent* set, so one stale print costs one point instead of many.

    Classic O(n log n) patience-sorting formulation.
    """
    v = np.asarray(values, dtype=float)
    if not increasing:
        v = -v
    n = v.size
    if n == 0:
        return np.empty(0, dtype=int)

    # tails[l] = index of the smallest possible tail of a subsequence of length l+1
    tails: list[int] = []
    prev = np.full(n, -1, dtype=int)

    for i in range(n):
        # bisect_right over the tail values -> allows equal values (non-strict)
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if v[tails[mid]] <= v[i]:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0:
            prev[i] = tails[lo - 1]
        if lo == len(tails):
            tails.append(i)
        else:
            tails[lo] = i

    out: list[int] = []
    cur = tails[-1]
    while cur != -1:
        out.append(cur)
        cur = prev[cur]
    return np.array(out[::-1], dtype=int)


def _resolve_prices(df: pd.DataFrame, cfg: Config, stats: dict[str, int]) -> pd.DataFrame:
    """Choose a price for every contract and record where it came from.

    Two-sided quotes win.  Outside market hours Yahoo withdraws most quotes,
    so a contract that traded in the most recent session is marked to its
    last trade instead -- which is exactly how an end-of-day surface is
    normally built.
    """
    df = df.copy()
    stats["raw_contracts"] = len(df)

    df = df.dropna(subset=["strike"])
    df = df[df["strike"] > 0]
    stats["after_valid_strike"] = len(df)

    bid = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
    ask = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
    last = pd.to_numeric(df["lastPrice"], errors="coerce").fillna(0.0)

    mid = 0.5 * (bid + ask)
    two_sided = (bid > 0) & (ask > 0) & (ask >= bid)
    rel_spread = np.where(two_sided & (mid > 0), (ask - bid) / mid.where(mid > 0, 1.0), np.nan)

    quote_ok = two_sided & (mid >= cfg.min_price) & (rel_spread <= cfg.max_rel_spread)
    stats["two_sided_quotes"] = int(two_sided.sum())
    stats["usable_quotes"] = int(quote_ok.sum())

    # Freshness: keep only trades from the most recent N sessions present in
    # the chain.  Deriving the session from the data itself means the filter
    # behaves correctly intraday, after the close, and on a cached snapshot.
    traded = pd.to_datetime(df["lastTradeDate"], errors="coerce", utc=True)
    trade_dates = traded.dt.date
    sessions = sorted({d for d in trade_dates.dropna().unique()})
    allowed = set(sessions[-cfg.max_trade_age_sessions:]) if sessions else set()
    fresh = trade_dates.isin(allowed)
    stats["fresh_trades"] = int(fresh.sum())

    last_ok = cfg.allow_last_trade & fresh & (last >= cfg.min_price)
    stats["usable_last_trades"] = int(last_ok.sum())

    df["price"] = np.where(quote_ok, mid, np.where(last_ok, last, np.nan))
    df["price_source"] = np.where(quote_ok, "mid", np.where(last_ok, "last", "none"))
    df["rel_spread"] = rel_spread
    df["last_traded"] = traded

    df = df[df["price_source"] != "none"]
    df = df[np.isfinite(df["price"]) & (df["price"] >= cfg.min_price)]
    stats["after_price_source"] = len(df)

    if sessions:
        log.info(
            "price sources: %d mid, %d last-trade (sessions kept: %s)",
            int((df["price_source"] == "mid").sum()),
            int((df["price_source"] == "last").sum()),
            ", ".join(str(s) for s in sorted(allowed)),
        )

    return df.reset_index(drop=True)


def _estimate_forward(
    slice_df: pd.DataFrame, T: float, rate: float, spot: float, dividend_yield: float
) -> tuple[float, str]:
    """Recover the forward for one expiry from put-call parity.

    Parity gives ``C - P = exp(-rT) (F - K)``, so every strike quoted on both
    sides yields ``F = K + exp(rT)(C - P)``.  We take the **median** over
    near-the-money strikes: near the money both legs are liquid, and the
    median is insensitive to the one or two stale prints that a chain of
    asynchronous last trades always contains.

    The estimate is accepted only if it lands within
    :data:`_MAX_FORWARD_DEVIATION` of the cost-of-carry forward.  A parity
    forward implying, say, a 3% dividend yield on SPY is a measurement
    failure, not a market view, and silently trusting it would shift the
    whole smile sideways.
    """
    carry = float(spot * np.exp((rate - dividend_yield) * T))

    calls = slice_df[slice_df["option_type"] == "call"].set_index("strike")["price"]
    puts = slice_df[slice_df["option_type"] == "put"].set_index("strike")["price"]
    calls = calls[~calls.index.duplicated()]
    puts = puts[~puts.index.duplicated()]
    common = calls.index.intersection(puts.index)

    if len(common) >= 3:
        strikes = np.asarray(common, dtype=float)
        near = np.abs(np.log(strikes / spot)) <= 0.15
        if near.sum() >= 3:
            diff = (calls.loc[common] - puts.loc[common]).to_numpy(dtype=float)
            estimates = strikes[near] + np.exp(rate * T) * diff[near]
            forward = float(np.median(estimates))
            if np.isfinite(forward) and forward > 0:
                if abs(np.log(forward / carry)) <= _MAX_FORWARD_DEVIATION:
                    return forward, "put_call_parity"
                log.debug(
                    "expiry T=%.3f: parity forward %.2f rejected vs carry %.2f", T, forward, carry
                )

    return carry, "carry"


def _select_otm(slice_df: pd.DataFrame, forward: float) -> pd.DataFrame:
    """Keep the out-of-the-money option at each strike.

    Puts below the forward, calls above.  OTM options are pure time value, so
    their prices carry the most volatility information per dollar of premium
    and their implied vols are by far the least sensitive to quote error.
    """
    strikes = slice_df["strike"].to_numpy(dtype=float)
    wanted = np.where(strikes < forward, "put", "call")
    picked = slice_df[slice_df["option_type"].to_numpy() == wanted]

    # One row per strike; if a strike somehow appears twice, prefer the
    # tighter quote and then the live mid over a last trade.
    picked = picked.sort_values(
        ["strike", "price_source", "rel_spread"], ascending=[True, True, True]
    )
    return picked.drop_duplicates(subset="strike", keep="first").reset_index(drop=True)


def _enforce_price_monotonicity(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Drop quotes that break monotonicity of price in strike.

    Static arbitrage requires call prices to fall and put prices to rise with
    strike.  A chain built from asynchronous last trades violates this
    routinely -- two adjacent strikes printed twenty minutes apart on a
    moving market.  Such pairs cannot both be right, and the resulting
    implied vols are wildly inconsistent, so we keep the largest mutually
    consistent subset on each side.
    """
    keep: list[pd.DataFrame] = []
    for side, increasing in (("put", True), ("call", False)):
        part = slice_df[slice_df["option_type"] == side].sort_values("strike")
        if len(part) <= 2:
            keep.append(part)
            continue
        idx = longest_monotone_subset(part["price"].to_numpy(dtype=float), increasing)
        keep.append(part.iloc[idx])
    return pd.concat(keep).sort_values("strike").reset_index(drop=True)


def _trim_to_sd_band(slice_df: pd.DataFrame, max_sd: float) -> pd.DataFrame:
    """Keep strikes within ``max_sd`` standard deviations of the forward.

    A fixed log-moneyness cutoff is the wrong shape: ``k = -0.3`` is a
    twenty-sigma lottery ticket on a one-week option and barely out of the
    money on a two-year one.  Scaling the band by ``iv * sqrt(T)`` -- the
    standard deviation of log-return to expiry -- keeps a comparable slice of
    the distribution at every maturity, which is both more defensible and
    much kinder to the calibration than trimming everything at one width.
    """
    if slice_df.empty:
        return slice_df

    T = float(slice_df["T"].iloc[0])
    k = slice_df["k"].to_numpy(dtype=float)
    iv = slice_df["iv"].to_numpy(dtype=float)

    # Anchor on the near-the-money vols, which are the reliable ones.
    near = np.argsort(np.abs(k))[: min(5, k.size)]
    iv_atm = float(np.median(iv[near]))
    if not np.isfinite(iv_atm) or iv_atm <= 0:
        return slice_df

    band = max_sd * iv_atm * np.sqrt(max(T, 1e-8))
    return slice_df[np.abs(k) <= band]


def _drop_smile_outliers(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Remove points far from a robust local quadratic fit of IV against k.

    A quadratic is a crude but serviceable smile model over a single expiry,
    so a point sitting many MADs off it is a bad print rather than genuine
    curvature.  Deliberately conservative: at 4 MADs this removes obvious
    breaks without shaving the wings the SVI fit is supposed to capture.
    """
    if len(slice_df) < 10:
        return slice_df

    k = slice_df["k"].to_numpy(dtype=float)
    iv = slice_df["iv"].to_numpy(dtype=float)
    try:
        coef = np.polyfit(k, iv, 2)
    except (np.linalg.LinAlgError, ValueError):  # pragma: no cover
        return slice_df

    resid = iv - np.polyval(coef, k)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    if mad <= 0:
        return slice_df

    keep = np.abs(resid - np.median(resid)) <= _OUTLIER_MADS * 1.4826 * mad
    return slice_df[keep]


def _select_expiries(candidates: list[str], max_expiries: int) -> list[str]:
    """Spread the expiry budget across the term structure, not the front end."""
    if len(candidates) <= max_expiries:
        return candidates
    idx = np.linspace(0, len(candidates) - 1, max_expiries).round().astype(int)
    return [candidates[i] for i in sorted(set(idx.tolist()))]


def build_surface(snap: ChainSnapshot, cfg: Config) -> SurfaceData:
    """Build a clean ``(k, T, IV)`` surface table from a raw chain snapshot."""
    stats: dict[str, int] = {}
    clean = _resolve_prices(snap.chain, cfg, stats)

    clean["T"] = [year_fraction(e, snap.asof) for e in clean["expiry"]]
    days = clean["T"] * _DAYS_PER_YEAR
    clean = clean[(days >= cfg.min_days) & (days <= cfg.max_days)]
    stats["after_expiry_window"] = len(clean)

    if clean.empty:
        raise RuntimeError(
            "no quotes survived the hygiene filters -- try --min-price 0.01 "
            "or a wider --min-days/--max-days window"
        )

    # Process every expiry in the window, then choose which fitted slices to
    # keep.  Selecting expiries up front would spend the budget on slices
    # that turn out to be unusable.
    per_expiry: dict[str, pd.DataFrame] = {}
    fwd_rows: dict[str, dict] = {}

    for expiry, slice_df in clean.groupby("expiry", sort=True):
        T = float(slice_df["T"].iloc[0])

        if cfg.imply_forward:
            forward, source = _estimate_forward(
                slice_df, T, cfg.rate, snap.spot, cfg.dividend_yield
            )
        else:
            forward = float(snap.spot * np.exp((cfg.rate - cfg.dividend_yield) * T))
            source = "carry"

        picked = _select_otm(slice_df, forward)
        if picked.empty:
            continue
        picked = _enforce_price_monotonicity(picked)
        if picked.empty:
            continue

        picked = picked.assign(
            forward=forward,
            forward_source=source,
            k=np.log(picked["strike"].to_numpy(dtype=float) / forward),
        )

        # Price off the forward: taking S_eff = F * exp(-rT) with q = 0 makes
        # the model's forward exactly the forward we measured, so the
        # inversion is consistent with the observed carry by construction.
        s_eff = forward * np.exp(-cfg.rate * T)
        iv, method, reason = implied_vol_vectorised(
            picked["price"].to_numpy(dtype=float),
            s_eff,
            picked["strike"].to_numpy(dtype=float),
            T,
            cfg.rate,
            0.0,
            picked["option_type"].to_numpy(dtype=object),
        )
        picked = picked.assign(
            iv=iv, iv_method=method, iv_reason=reason, spot_effective=s_eff
        )

        per_expiry[expiry] = picked
        fwd_rows[expiry] = {
            "expiry": expiry,
            "T": T,
            "days": T * _DAYS_PER_YEAR,
            "forward": forward,
            "forward_source": source,
        }

    if not per_expiry:
        raise RuntimeError("no expiry produced any usable out-of-the-money quotes")

    points = pd.concat(per_expiry.values(), ignore_index=True)
    stats["after_otm_and_monotone"] = len(points)
    stats["iv_solved"] = int(np.isfinite(points["iv"]).sum())

    points = points[np.isfinite(points["iv"])]
    points = points[(points["iv"] >= IV_MIN) & (points["iv"] <= IV_MAX)]
    stats["after_iv_level"] = len(points)

    points = points[points["k"].abs() <= cfg.max_abs_log_moneyness]
    stats["after_moneyness"] = len(points)

    points["vega"] = bs_vega(
        points["spot_effective"].to_numpy(dtype=float),
        points["strike"].to_numpy(dtype=float),
        points["T"].to_numpy(dtype=float),
        cfg.rate,
        0.0,
        points["iv"].to_numpy(dtype=float),
    )
    points = points[points["vega"] > _MIN_VEGA]
    stats["after_vega"] = len(points)

    points = pd.concat(
        [
            _trim_to_sd_band(grp, cfg.max_sd_moneyness)
            for _, grp in points.groupby("expiry", sort=False)
        ],
        ignore_index=True,
    )
    stats["after_sd_band"] = len(points)

    points = pd.concat(
        [_drop_smile_outliers(grp) for _, grp in points.groupby("expiry", sort=False)],
        ignore_index=True,
    )
    stats["after_outliers"] = len(points)

    # A five-parameter slice needs enough points, and needs them on both
    # sides of the money: a one-winged slice cannot identify rho and m.
    good: list[str] = []
    for expiry, grp in points.groupby("expiry"):
        n_left = int((grp["k"] < 0).sum())
        n_right = int((grp["k"] > 0).sum())
        if (
            len(grp) >= cfg.min_points_per_slice
            and n_left >= cfg.min_points_per_wing
            and n_right >= cfg.min_points_per_wing
        ):
            good.append(expiry)
        else:
            log.debug(
                "dropping %s: n=%d (left %d, right %d)", expiry, len(grp), n_left, n_right
            )

    if not good:
        raise RuntimeError(
            "no expiry slice has enough clean quotes on both wings to support a fit"
        )

    stats["n_candidate_expiries"] = len(good)
    selected = _select_expiries(sorted(good), cfg.max_expiries)
    points = points[points["expiry"].isin(selected)]
    stats["after_thin_slices"] = len(points)
    stats["n_expiries"] = len(selected)

    points["total_var"] = points["iv"] ** 2 * points["T"]
    points = points.sort_values(["T", "k"]).reset_index(drop=True)

    forwards = pd.DataFrame([fwd_rows[e] for e in selected])
    forwards["n_points"] = forwards["expiry"].map(points.groupby("expiry").size())
    forwards = forwards.sort_values("T").reset_index(drop=True)

    mix = points["price_source"].value_counts().to_dict()

    log.info(
        "surface: %d points across %d expiries (from %d raw contracts); price mix %s",
        len(points), len(selected), stats["raw_contracts"], mix,
    )

    return SurfaceData(
        points=points,
        forwards=forwards,
        spot=snap.spot,
        asof=snap.asof,
        ticker=snap.ticker,
        rate=cfg.rate,
        dividend_yield=cfg.dividend_yield,
        filter_stats=stats,
        price_source_mix={str(k): int(v) for k, v in mix.items()},
        config=cfg,
    )
