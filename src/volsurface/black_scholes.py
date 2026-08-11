"""Black-Scholes-Merton pricing and greeks, implemented from scratch.

Everything here is written against the standard BSM model with a continuous
dividend yield ``q``.  Functions are vectorised over numpy arrays and are the
single source of truth for pricing in the rest of the package -- no external
option-pricing library is used anywhere.

Convention throughout the package:

* ``S``     spot price of the underlying
* ``K``     strike
* ``T``     time to expiry in years (ACT/365)
* ``r``     continuously compounded risk-free rate
* ``q``     continuous dividend yield
* ``sigma`` Black-Scholes volatility (annualised, as a decimal, e.g. 0.20)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = [
    "OptionType",
    "norm_cdf",
    "norm_pdf",
    "d1_d2",
    "bs_price",
    "bs_call_price",
    "bs_put_price",
    "bs_vega",
    "forward_price",
    "intrinsic_value",
    "price_bounds",
]

OptionType = Literal["call", "put"]

# 1/sqrt(2*pi), precomputed for the normal density.
_INV_SQRT_2PI = 0.3989422804014326779399460599343818684758586311649


def norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF via the error function.

    ``scipy.special.ndtr`` would do the same thing, but the point of this
    project is that the pricing stack is self-contained; ``math.erf`` is a
    C-library primitive, not an option-pricing routine.
    """
    from scipy.special import erf  # local import keeps module import cheap

    return 0.5 * (1.0 + erf(np.asarray(x, dtype=float) / np.sqrt(2.0)))


def norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal PDF."""
    x = np.asarray(x, dtype=float)
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def forward_price(S, T, r: float, q: float = 0.0):
    """Forward price ``F = S * exp((r - q) * T)``."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    return S * np.exp((r - q) * T)


def d1_d2(S, K, T, r: float, q: float, sigma):
    """Return the Black-Scholes ``d1`` and ``d2`` terms.

    Degenerate inputs (``T <= 0`` or ``sigma <= 0``) produce +/- inf so that
    the pricing formulas collapse to the discounted intrinsic value rather
    than raising or returning NaN.
    """
    S, K, T, sigma = (np.asarray(v, dtype=float) for v in (S, K, T, sigma))

    vol_sqrt_t = sigma * np.sqrt(np.maximum(T, 0.0))
    degenerate = (vol_sqrt_t <= 0.0) | (S <= 0.0) | (K <= 0.0)

    # Use a safe denominator so numpy does not emit divide-by-zero warnings;
    # the degenerate entries are overwritten immediately below.
    safe_den = np.where(degenerate, 1.0, vol_sqrt_t)
    safe_S = np.where(S <= 0.0, 1.0, S)
    safe_K = np.where(K <= 0.0, 1.0, K)

    log_moneyness = np.log(safe_S / safe_K)
    d1 = (log_moneyness + (r - q + 0.5 * sigma**2) * np.maximum(T, 0.0)) / safe_den
    d2 = d1 - safe_den

    # In the zero-variance limit the option is worth its (discounted)
    # intrinsic value, which corresponds to d1 = d2 = +/- inf.
    fwd_log = log_moneyness + (r - q) * np.maximum(T, 0.0)
    limit = np.where(fwd_log > 0.0, np.inf, np.where(fwd_log < 0.0, -np.inf, 0.0))
    d1 = np.where(degenerate, limit, d1)
    d2 = np.where(degenerate, limit, d2)
    return d1, d2


def bs_call_price(S, K, T, r: float, q: float, sigma):
    """Black-Scholes price of a European call."""
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    S, K, T = (np.asarray(v, dtype=float) for v in (S, K, T))
    T = np.maximum(T, 0.0)
    price = S * np.exp(-q * T) * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    return np.maximum(price, 0.0)


def bs_put_price(S, K, T, r: float, q: float, sigma):
    """Black-Scholes price of a European put."""
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    S, K, T = (np.asarray(v, dtype=float) for v in (S, K, T))
    T = np.maximum(T, 0.0)
    price = K * np.exp(-r * T) * norm_cdf(-d2) - S * np.exp(-q * T) * norm_cdf(-d1)
    return np.maximum(price, 0.0)


def bs_price(S, K, T, r: float, q: float, sigma, option_type: OptionType):
    """Dispatch to :func:`bs_call_price` / :func:`bs_put_price`."""
    if option_type == "call":
        return bs_call_price(S, K, T, r, q, sigma)
    if option_type == "put":
        return bs_put_price(S, K, T, r, q, sigma)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_vega(S, K, T, r: float, q: float, sigma):
    """Vega -- dPrice/dSigma, per 1.0 (i.e. 100 vol points) change in sigma.

    Identical for calls and puts.  Divide by 100 for the "per vol point"
    convention used on trading desks.
    """
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    S, T = np.asarray(S, dtype=float), np.maximum(np.asarray(T, dtype=float), 0.0)
    vega = S * np.exp(-q * T) * norm_pdf(d1) * np.sqrt(T)
    # d1 is +/-inf in the degenerate case; pdf underflows to 0, which is right.
    return np.nan_to_num(vega, nan=0.0, posinf=0.0, neginf=0.0)


def intrinsic_value(S, K, T, r: float, q: float, option_type: OptionType):
    """Discounted intrinsic value -- the zero-volatility price, and the
    lower no-arbitrage bound for a European option."""
    S, K, T = (np.asarray(v, dtype=float) for v in (S, K, T))
    T = np.maximum(T, 0.0)
    fwd_pv = S * np.exp(-q * T)
    strike_pv = K * np.exp(-r * T)
    if option_type == "call":
        return np.maximum(fwd_pv - strike_pv, 0.0)
    if option_type == "put":
        return np.maximum(strike_pv - fwd_pv, 0.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def price_bounds(S, K, T, r: float, q: float, option_type: OptionType):
    """``(lower, upper)`` no-arbitrage price bounds for a European option.

    A quote outside these bounds cannot be matched by *any* non-negative
    volatility, so the implied-vol solver rejects it rather than returning a
    boundary artefact.
    """
    S, K, T = (np.asarray(v, dtype=float) for v in (S, K, T))
    T = np.maximum(T, 0.0)
    lower = intrinsic_value(S, K, T, r, q, option_type)
    upper = S * np.exp(-q * T) if option_type == "call" else K * np.exp(-r * T)
    return lower, np.broadcast_to(np.asarray(upper, dtype=float), np.shape(lower)).copy()
