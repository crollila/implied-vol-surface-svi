"""Robust inversion of the Black-Scholes formula for implied volatility.

Strategy, per option:

1. Reject quotes that violate the static no-arbitrage price bounds -- no
   volatility can reproduce them, so any "solution" would be an artefact of
   the solver hitting a bracket edge.
2. Bracket the root by expanding upward from a small volatility until the
   model price exceeds the market price, then run **Brent's method** on the
   price residual.  Brent is derivative-free and guaranteed to converge on a
   bracketed sign change, which makes it the right primary method for noisy
   market quotes.
3. If bracketing fails (typically deep wings where the price is flat in
   sigma to machine precision), fall back to a safeguarded **Newton**
   iteration using analytic vega, with a bisection guard so a tiny vega
   cannot throw the iterate off to nonsense.

Every failure mode returns ``nan`` plus a machine-readable reason rather than
raising, so a single bad row never kills a pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .black_scholes import OptionType, bs_price, bs_vega, price_bounds

__all__ = ["IVResult", "implied_vol", "implied_vol_vectorised"]

# Volatility search domain.  1000% vol is far beyond anything a real listed
# equity-index option trades at; anything hitting the ceiling is a bad quote.
VOL_LOWER = 1e-6
VOL_UPPER = 10.0

# Price tolerance used when checking against the no-arbitrage bounds.  Quotes
# are penny-quoted at best, so a sub-basis-point cushion only ever removes
# genuinely inconsistent rows.
_BOUND_TOL = 1e-10


@dataclass(frozen=True)
class IVResult:
    """Outcome of one implied-vol inversion."""

    iv: float
    """Implied volatility, or ``nan`` if the inversion failed."""

    method: str
    """Which solver produced the answer: ``brent``, ``newton`` or ``failed``."""

    reason: str = ""
    """Machine-readable failure reason; empty on success."""

    @property
    def ok(self) -> bool:
        return np.isfinite(self.iv)


def _price_residual(sigma: float, price: float, S, K, T, r, q, option_type) -> float:
    return float(bs_price(S, K, T, r, q, sigma, option_type)) - price


def _newton(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    x0: float = 0.25,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Safeguarded Newton iteration on the price residual.

    Maintains a bracket alongside the Newton iterate; whenever a Newton step
    would leave the bracket (or vega is too small to trust) it falls back to a
    bisection step.  This keeps the quadratic convergence of Newton where the
    problem is well conditioned without inheriting its divergence on the
    flat wings.
    """
    lo, hi = VOL_LOWER, VOL_UPPER
    x = float(np.clip(x0, lo, hi))

    for _ in range(max_iter):
        f = _price_residual(x, price, S, K, T, r, q, option_type)
        if abs(f) < tol:
            return x

        # Price is increasing in sigma, so the sign of the residual tells us
        # which side of the root we are on.
        if f > 0.0:
            hi = x
        else:
            lo = x

        v = float(bs_vega(S, K, T, r, q, x))
        if v > 1e-12:
            step = f / v
            x_new = x - step
        else:
            x_new = np.nan

        # Reject a Newton step that escapes the bracket or is not finite.
        if not np.isfinite(x_new) or not (lo < x_new < hi):
            x_new = 0.5 * (lo + hi)

        if abs(x_new - x) < 1e-14:
            return x_new
        x = x_new

    return x if abs(_price_residual(x, price, S, K, T, r, q, option_type)) < 1e-6 else np.nan


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
) -> IVResult:
    """Invert Black-Scholes for a single option quote.

    Parameters
    ----------
    price:
        Observed option price (typically the bid/ask mid).
    S, K, T, r, q:
        Spot, strike, year fraction to expiry, risk-free rate, dividend yield.
    option_type:
        ``"call"`` or ``"put"``.

    Returns
    -------
    IVResult
        ``iv`` is ``nan`` when the quote is unusable; ``reason`` says why.
    """
    if not all(np.isfinite([price, S, K, T])):
        return IVResult(np.nan, "failed", "non_finite_input")
    if T <= 0.0:
        return IVResult(np.nan, "failed", "expired")
    if S <= 0.0 or K <= 0.0:
        return IVResult(np.nan, "failed", "non_positive_spot_or_strike")
    if price <= 0.0:
        return IVResult(np.nan, "failed", "non_positive_price")

    lower, upper = price_bounds(S, K, T, r, q, option_type)
    lower, upper = float(lower), float(upper)

    if price < lower - _BOUND_TOL:
        # Below discounted intrinsic: negative time value, an arbitrage-y or
        # stale quote.
        return IVResult(np.nan, "failed", "below_intrinsic")
    if price >= upper - _BOUND_TOL:
        # At or above the asymptotic price ceiling: implies infinite vol.
        return IVResult(np.nan, "failed", "above_upper_bound")
    if price - lower < 1e-8:
        # Time value is numerically zero -- the price is flat in sigma here,
        # so any reported IV would be noise.
        return IVResult(np.nan, "failed", "zero_time_value")

    # --- Step 1: bracket the root by expanding the upper edge. ------------
    f_lo = _price_residual(VOL_LOWER, price, S, K, T, r, q, option_type)
    if f_lo > 0.0:
        # Even a vanishing vol overprices the quote; already covered by the
        # intrinsic check, but keep the guard for pathological inputs.
        return IVResult(np.nan, "failed", "below_intrinsic")

    hi = 0.5
    f_hi = _price_residual(hi, price, S, K, T, r, q, option_type)
    while f_hi < 0.0 and hi < VOL_UPPER:
        hi = min(hi * 2.0, VOL_UPPER)
        f_hi = _price_residual(hi, price, S, K, T, r, q, option_type)

    # --- Step 2: Brent on the bracket. ------------------------------------
    if f_hi >= 0.0:
        try:
            iv = brentq(
                _price_residual,
                VOL_LOWER,
                hi,
                args=(price, S, K, T, r, q, option_type),
                xtol=1e-12,
                rtol=8.9e-16,
                maxiter=200,
            )
            if np.isfinite(iv) and VOL_LOWER < iv < VOL_UPPER:
                return IVResult(float(iv), "brent")
        except (ValueError, RuntimeError):
            pass  # fall through to Newton

    # --- Step 3: Newton fallback. -----------------------------------------
    iv = _newton(price, S, K, T, r, q, option_type)
    if np.isfinite(iv) and VOL_LOWER < iv < VOL_UPPER:
        return IVResult(float(iv), "newton")

    return IVResult(np.nan, "failed", "no_convergence")


def implied_vol_vectorised(
    prices,
    S,
    K,
    T,
    r: float,
    q: float,
    option_types,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply :func:`implied_vol` elementwise.

    Returns ``(iv, method, reason)`` as three aligned object/float arrays.
    Loop-based by necessity: Brent is a scalar root finder, and at a few
    thousand quotes per chain the cost is irrelevant next to the network
    fetch.
    """
    prices, S_arr, K_arr, T_arr = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (prices, S, K, T))
    types = np.atleast_1d(np.asarray(option_types, dtype=object))

    n = max(len(prices), len(K_arr), len(T_arr), len(types))
    S_arr, K_arr, T_arr, prices, types = (
        np.broadcast_to(a, (n,)) for a in (S_arr, K_arr, T_arr, prices, types)
    )

    iv = np.full(n, np.nan)
    method = np.empty(n, dtype=object)
    reason = np.empty(n, dtype=object)

    for i in range(n):
        res = implied_vol(prices[i], S_arr[i], K_arr[i], T_arr[i], r, q, types[i])
        iv[i] = res.iv
        method[i] = res.method
        reason[i] = res.reason

    return iv, method, reason
