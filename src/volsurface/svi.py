"""Raw SVI parameterisation, calibration, and arbitrage diagnostics.

The raw SVI form (Gatheral, 2004) models **total implied variance**
``w = sigma_BS^2 * T`` as a function of log-moneyness ``k = log(K / F)``:

.. math::

    w(k) = a + b \\left( \\rho (k - m) + \\sqrt{(k - m)^2 + \\sigma^2} \\right)

with the five parameters carrying direct financial meaning:

======  ====================================================================
``a``   vertical level -- roughly the ATM total variance floor
``b``   overall slope / wing steepness (angle between the asymptotes)
``rho`` skew: negative tilts the smile down-and-right, the equity-index norm
``m``   horizontal shift of the smile's minimum
``sig`` curvature at the minimum: ``sigma -> 0`` gives a kinked V
======  ====================================================================

Asymptotically ``w(k) ~ a + b(1 + rho)(k - m)`` as ``k -> +inf`` and
``a - b(1 - rho)(k - m)`` as ``k -> -inf``, so the wing slopes are
``b(1 +/- rho)``.  Roger Lee's moment formula caps those slopes at 2, which
is the economic content of the ``b`` bound used below.

Calibration is a bounded nonlinear least-squares fit per expiry slice with
multi-start initialisation, followed by a Durrleman butterfly-arbitrage
check; slices that fail can be re-fit with the arbitrage condition as a
penalty.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

log = logging.getLogger(__name__)

__all__ = [
    "SVIParams",
    "SVIFit",
    "svi_total_variance",
    "svi_derivatives",
    "svi_implied_vol",
    "durrleman_g",
    "check_butterfly",
    "fit_slice",
    "check_calendar_arbitrage",
]

#: Lee's moment formula bounds the wing slopes b(1 +/- rho) by 2.
_MAX_WING_SLOPE = 2.0

#: Tolerance for declaring a Durrleman violation.  Small negative values on
#: a discrete grid are numerical, not economic.
_BUTTERFLY_TOL = -1e-8

#: How many of the multi-start solutions get polished to full precision.
_N_POLISH = 3


@dataclass(frozen=True)
class SVIParams:
    """The five raw-SVI parameters."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma], dtype=float)

    @classmethod
    def from_array(cls, x) -> "SVIParams":
        a, b, rho, m, sigma = (float(v) for v in x)
        return cls(a, b, rho, m, sigma)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def min_total_variance(self) -> float:
        """The minimum of ``w(k)`` over all ``k``.

        Attained at ``k = m - rho*sigma/sqrt(1-rho^2)``; must be >= 0 for the
        slice to be a legitimate variance curve.
        """
        return self.a + self.b * self.sigma * np.sqrt(max(1.0 - self.rho**2, 0.0))

    @property
    def wing_slopes(self) -> tuple[float, float]:
        """``(left, right)`` asymptotic slopes of ``w`` in ``|k|``."""
        return self.b * (1.0 - self.rho), self.b * (1.0 + self.rho)


@dataclass
class SVIFit:
    """Result of calibrating one expiry slice."""

    expiry: str
    T: float
    params: SVIParams
    n_points: int
    rmse_total_var: float
    rmse_iv: float
    """RMSE in Black-Scholes vol units (decimals, so 0.0042 = 42 bps)."""
    max_abs_err_iv: float
    r_squared: float
    butterfly_ok: bool
    min_durrleman_g: float
    min_total_variance: float
    k_min: float
    k_max: float
    converged: bool
    n_starts_used: int
    penalised: bool
    """True if the slice needed the arbitrage-penalised refit."""

    def to_row(self) -> dict:
        p = self.params
        left, right = p.wing_slopes
        return {
            "expiry": self.expiry,
            "T": round(self.T, 6),
            "days": round(self.T * 365.0, 2),
            "n_points": self.n_points,
            "a": p.a,
            "b": p.b,
            "rho": p.rho,
            "m": p.m,
            "sigma": p.sigma,
            "rmse_total_var": self.rmse_total_var,
            "rmse_iv": self.rmse_iv,
            "rmse_iv_bps": self.rmse_iv * 1e4,
            "max_abs_err_iv_bps": self.max_abs_err_iv * 1e4,
            "r_squared": self.r_squared,
            "atm_total_var": float(svi_total_variance(0.0, p)),
            "atm_iv": float(svi_implied_vol(0.0, self.T, p)),
            "wing_slope_left": left,
            "wing_slope_right": right,
            "min_total_variance": self.min_total_variance,
            "butterfly_ok": self.butterfly_ok,
            "min_durrleman_g": self.min_durrleman_g,
            "penalised_refit": self.penalised,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "converged": self.converged,
        }


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def svi_total_variance(k, params: SVIParams):
    """Total implied variance ``w(k)`` under raw SVI."""
    k = np.asarray(k, dtype=float)
    x = k - params.m
    return params.a + params.b * (params.rho * x + np.sqrt(x * x + params.sigma**2))


def svi_derivatives(k, params: SVIParams) -> tuple[np.ndarray, np.ndarray]:
    """First and second derivatives of ``w`` with respect to ``k``.

    ``w'(k)  = b (rho + x/s)`` and ``w''(k) = b sigma^2 / s^3``,
    where ``x = k - m`` and ``s = sqrt(x^2 + sigma^2)``.
    """
    k = np.asarray(k, dtype=float)
    x = k - params.m
    s = np.sqrt(x * x + params.sigma**2)
    s = np.where(s <= 0.0, 1e-16, s)
    w1 = params.b * (params.rho + x / s)
    w2 = params.b * params.sigma**2 / s**3
    return w1, w2


def svi_implied_vol(k, T: float, params: SVIParams):
    """Black-Scholes implied vol implied by the slice: ``sqrt(w(k) / T)``."""
    w = np.maximum(svi_total_variance(k, params), 0.0)
    return np.sqrt(w / T)


def durrleman_g(k, params: SVIParams):
    """Durrleman's function ``g(k)``.

    The risk-neutral density implied by a slice is proportional to
    ``g(k)``; ``g(k) >= 0`` for all ``k`` is exactly the statement that the
    slice admits no **butterfly arbitrage** (no negative-probability region).

    .. math::

        g(k) = \\left(1 - \\frac{k w'(k)}{2 w(k)}\\right)^2
               - \\frac{w'(k)^2}{4}\\left(\\frac{1}{w(k)} + \\frac14\\right)
               + \\frac{w''(k)}{2}
    """
    k = np.asarray(k, dtype=float)
    w = svi_total_variance(k, params)
    w1, w2 = svi_derivatives(k, params)
    w_safe = np.where(w <= 1e-12, 1e-12, w)

    term1 = (1.0 - k * w1 / (2.0 * w_safe)) ** 2
    term2 = (w1**2 / 4.0) * (1.0 / w_safe + 0.25)
    return term1 - term2 + w2 / 2.0


def check_butterfly(
    params: SVIParams, k_lo: float = -2.0, k_hi: float = 2.0, n: int = 1001
) -> tuple[bool, float]:
    """Scan ``g(k)`` on a dense grid.  Returns ``(is_arbitrage_free, min_g)``.

    A grid scan rather than an analytic condition: the closed-form conditions
    for raw SVI are sufficient but conservative, and the grid tells us *how*
    badly a slice fails, which is more useful diagnostically.
    """
    grid = np.linspace(k_lo, k_hi, n)
    g = durrleman_g(grid, params)
    min_g = float(np.min(g))
    if params.min_total_variance < 0.0:
        return False, min_g
    return bool(min_g >= _BUTTERFLY_TOL), min_g


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _bounds(k: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w_max = float(np.max(w))
    k_lo, k_hi = float(np.min(k)), float(np.max(k))
    span = max(k_hi - k_lo, 0.1)

    lower = np.array([-w_max, 1e-8, -0.9999, k_lo - span, 1e-6])
    upper = np.array([2.0 * w_max, _MAX_WING_SLOPE, 0.9999, k_hi + span, 2.0 * span + 1.0])
    return lower, upper


def _initial_guesses(
    k: np.ndarray, w: np.ndarray, n_starts: int, seed: int
) -> list[np.ndarray]:
    """Deterministic heuristic starts, topped up with seeded random ones."""
    w_min, w_max = float(np.min(w)), float(np.max(w))
    k_at_min = float(k[int(np.argmin(w))])
    span = max(float(np.max(k) - np.min(k)), 0.1)
    slope_scale = max((w_max - w_min) / span, 1e-3)

    starts: list[np.ndarray] = []
    for rho0 in (-0.85, -0.6, -0.3, 0.0, 0.3):
        for sigma0 in (0.05, 0.2, 0.5):
            for m0 in (k_at_min, 0.0):
                b0 = float(np.clip(slope_scale, 1e-3, _MAX_WING_SLOPE * 0.9))
                a0 = max(w_min * 0.5, 1e-8)
                starts.append(np.array([a0, b0, rho0, m0, sigma0]))

    rng = np.random.default_rng(seed)
    while len(starts) < n_starts:
        starts.append(
            np.array(
                [
                    rng.uniform(0.0, w_max),
                    rng.uniform(1e-3, 1.0),
                    rng.uniform(-0.95, 0.5),
                    rng.uniform(float(np.min(k)), float(np.max(k))),
                    rng.uniform(0.01, 1.0),
                ]
            )
        )

    return starts[:n_starts]


#: The penalised refit targets ``g(k) >= _BUTTERFLY_MARGIN`` rather than
#: ``g(k) >= 0``, so the solution lands strictly inside the no-arbitrage
#: region instead of resting exactly on its boundary, where grid resolution
#: alone decides whether the check passes.
_BUTTERFLY_MARGIN = 1e-4

#: Weight on the negative-variance guard.  Stiff enough to be effectively a
#: hard constraint, smooth enough for a gradient-based solver.
_NEG_VAR_WEIGHT = 100.0


def _residual_factory(
    k: np.ndarray,
    w: np.ndarray,
    weights: np.ndarray,
    butterfly_penalty: float,
    penalty_grid: np.ndarray | None,
):
    """Build the residual vector for :func:`scipy.optimize.least_squares`."""
    sqrt_w = np.sqrt(weights)

    def residuals(x: np.ndarray) -> np.ndarray:
        params = SVIParams.from_array(x)
        model = svi_total_variance(k, params)
        res = sqrt_w * (model - w)

        # Hard requirement: the fitted curve is a variance, so it must be
        # non-negative everywhere.
        min_w = params.min_total_variance
        neg = np.array([_NEG_VAR_WEIGHT * min(min_w, 0.0)])

        if butterfly_penalty > 0.0 and penalty_grid is not None:
            g = durrleman_g(penalty_grid, params)
            viol = np.minimum(g - _BUTTERFLY_MARGIN, 0.0)
            return np.concatenate([res, neg, butterfly_penalty * viol])

        return np.concatenate([res, neg])

    return residuals


def _jacobian_factory(k: np.ndarray, weights: np.ndarray):
    """Analytic Jacobian of the unpenalised residual vector.

    Differentiating the raw SVI form with ``x = k - m`` and
    ``s = sqrt(x^2 + sigma^2)``::

        dw/da     = 1
        dw/db     = rho*x + s
        dw/drho   = b*x
        dw/dm     = -b*(rho + x/s)
        dw/dsigma = b*sigma/s

    Supplying this instead of letting SciPy build a 2-point finite
    difference cuts the number of residual evaluations per iteration by
    roughly six, which matters because the calibration is multi-start.
    """
    sqrt_w = np.sqrt(weights)
    n = k.size

    def jac(x: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = (float(v) for v in x)
        xx = k - m
        s = np.sqrt(xx * xx + sigma**2)
        s = np.where(s <= 0.0, 1e-16, s)

        j = np.empty((n + 1, 5))
        j[:n, 0] = sqrt_w
        j[:n, 1] = sqrt_w * (rho * xx + s)
        j[:n, 2] = sqrt_w * (b * xx)
        j[:n, 3] = sqrt_w * (-b * (rho + xx / s))
        j[:n, 4] = sqrt_w * (b * sigma / s)

        # Row for the negative-variance guard:
        # min_w = a + b*sigma*sqrt(1 - rho^2), active only when it is < 0.
        root = np.sqrt(max(1.0 - rho * rho, 1e-16))
        if a + b * sigma * root < 0.0:
            j[n, 0] = _NEG_VAR_WEIGHT
            j[n, 1] = _NEG_VAR_WEIGHT * sigma * root
            j[n, 2] = _NEG_VAR_WEIGHT * b * sigma * (-rho / root)
            j[n, 3] = 0.0
            j[n, 4] = _NEG_VAR_WEIGHT * b * root
        else:
            j[n, :] = 0.0

        return j

    return jac


def _solve(
    k: np.ndarray,
    w: np.ndarray,
    weights: np.ndarray,
    starts: list[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    butterfly_penalty: float = 0.0,
    penalty_grid: np.ndarray | None = None,
) -> tuple[SVIParams | None, bool, int]:
    """Run multi-start least squares; return the best solution found.

    Two stages.  Every start gets a cheap, loosely converged run; only the
    handful with the lowest cost are then polished to full precision.  Bad
    starting points are identified as bad long before machine precision, so
    grinding all of them down is wasted work -- this is several times faster
    than polishing everything, with the same answer.
    """
    residuals = _residual_factory(k, w, weights, butterfly_penalty, penalty_grid)
    # The analytic Jacobian covers the plain residual only; the butterfly
    # penalty path is rare enough to leave on finite differences.
    jac = "2-point" if butterfly_penalty > 0.0 else _jacobian_factory(k, weights)

    def run(x0: np.ndarray, tol: float, max_nfev: int):
        return least_squares(
            residuals,
            np.clip(x0, lower + 1e-12, upper - 1e-12),
            jac=jac,
            bounds=(lower, upper),
            method="trf",
            loss="linear",
            xtol=tol,
            ftol=tol,
            gtol=tol,
            max_nfev=max_nfev,
        )

    coarse: list[tuple[float, np.ndarray]] = []
    used = 0
    for x0 in starts:
        try:
            sol = run(x0, 1e-8, 400)
        except Exception:  # pragma: no cover - solver blow-up on a bad start
            continue
        used += 1
        coarse.append((float(sol.cost), sol.x.copy()))

    if not coarse:
        return None, False, used

    coarse.sort(key=lambda t: t[0])

    best_x: np.ndarray | None = None
    best_cost = np.inf
    best_ok = False
    for _, x0 in coarse[:_N_POLISH]:
        try:
            sol = run(x0, 1e-14, 5000)
        except Exception:  # pragma: no cover
            continue
        if sol.cost < best_cost:
            best_cost = float(sol.cost)
            best_x = sol.x.copy()
            best_ok = bool(sol.success)

    if best_x is None:
        best_cost, best_x = coarse[0]
        best_ok = False

    return SVIParams.from_array(best_x), best_ok, used


def fit_slice(
    k,
    total_var,
    T: float,
    expiry: str = "",
    weights=None,
    n_starts: int = 24,
    seed: int = 7,
    enforce_no_butterfly: bool = True,
) -> SVIFit:
    """Calibrate raw SVI to one expiry slice.

    Parameters
    ----------
    k:
        Log-moneyness ``log(K / F)`` of each observation.
    total_var:
        Observed total implied variance ``iv**2 * T``.
    T:
        Year fraction to expiry, used to convert fit errors back to vol units.
    weights:
        Optional per-point least-squares weights (vega weighting in the
        pipeline).  Normalised internally, so only relative size matters.
    enforce_no_butterfly:
        If the unconstrained fit admits butterfly arbitrage, refit with
        Durrleman's condition as a penalty and keep the arbitrage-free
        solution.

    Returns
    -------
    SVIFit
        Fitted parameters together with fit-quality and arbitrage diagnostics.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_var, dtype=float)

    if k.shape != w.shape:
        raise ValueError("k and total_var must have the same shape")
    if k.size < 5:
        raise ValueError(f"need at least 5 points to fit 5 SVI parameters, got {k.size}")

    if weights is None:
        weights = np.ones_like(w)
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(w)
    weights = weights / weights.mean()

    lower, upper = _bounds(k, w)
    starts = _initial_guesses(k, w, n_starts, seed)

    params, converged, used = _solve(k, w, weights, starts, lower, upper)
    if params is None:
        raise RuntimeError(f"SVI calibration failed for expiry {expiry!r}: no solver run succeeded")

    # Check the fitted slice a little beyond the quoted strike range: an
    # extrapolated arbitrage still matters if anyone prices off the surface.
    pad = 0.5 * max(float(k.max() - k.min()), 0.2)
    g_lo, g_hi = float(k.min()) - pad, float(k.max()) + pad
    ok, min_g = check_butterfly(params, g_lo, g_hi)

    penalised = False
    if enforce_no_butterfly and not ok:
        grid = np.linspace(g_lo, g_hi, 201)
        p2, conv2, used2 = _solve(
            k, w, weights, [params.as_array()] + starts, lower, upper,
            butterfly_penalty=50.0, penalty_grid=grid,
        )
        if p2 is not None:
            ok2, min_g2 = check_butterfly(p2, g_lo, g_hi)
            if min_g2 > min_g:
                params, converged, ok, min_g = p2, conv2, ok2, min_g2
                used += used2
                penalised = True
                log.debug("expiry %s: refit with butterfly penalty (min g %.2e)", expiry, min_g)

    model_w = svi_total_variance(k, params)
    resid_w = model_w - w
    rmse_w = float(np.sqrt(np.mean(resid_w**2)))

    model_iv = np.sqrt(np.maximum(model_w, 0.0) / T)
    market_iv = np.sqrt(np.maximum(w, 0.0) / T)
    resid_iv = model_iv - market_iv
    rmse_iv = float(np.sqrt(np.mean(resid_iv**2)))

    ss_res = float(np.sum(resid_w**2))
    ss_tot = float(np.sum((w - w.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return SVIFit(
        expiry=expiry,
        T=T,
        params=params,
        n_points=int(k.size),
        rmse_total_var=rmse_w,
        rmse_iv=rmse_iv,
        max_abs_err_iv=float(np.max(np.abs(resid_iv))),
        r_squared=r2,
        butterfly_ok=ok,
        min_durrleman_g=min_g,
        min_total_variance=float(params.min_total_variance),
        k_min=float(k.min()),
        k_max=float(k.max()),
        converged=converged,
        n_starts_used=used,
        penalised=penalised,
    )


def check_calendar_arbitrage(
    fits: list[SVIFit],
    k_lo: float | None = None,
    k_hi: float | None = None,
    n: int = 201,
) -> tuple[bool, list[dict]]:
    """Test that total variance is non-decreasing in maturity at fixed ``k``.

    Calendar arbitrage exists if two slices cross: buying the cheaper,
    longer-dated total variance and selling the dearer, shorter-dated one is
    a riskless profit.  Formally the surface must satisfy
    ``w(k, T1) <= w(k, T2)`` whenever ``T1 < T2``.

    By default each adjacent pair is tested only on the **overlap of the two
    slices' fitted moneyness ranges**.  That restriction matters: a one-month
    slice quoted over ``k in [-0.15, 0.13]`` says nothing about ``k = 0.5``,
    and comparing two SVI curves where both are extrapolating measures the
    parameterisation's tail behaviour rather than any tradeable arbitrage.
    Pass explicit ``k_lo`` / ``k_hi`` to force a fixed test band instead.

    Returns ``(is_arbitrage_free, violations)``; each violation records the
    adjacent pair and the worst crossing found.
    """
    ordered = sorted(fits, key=lambda f: f.T)
    violations: list[dict] = []

    for near, far in zip(ordered, ordered[1:]):
        lo = k_lo if k_lo is not None else max(near.k_min, far.k_min)
        hi = k_hi if k_hi is not None else min(near.k_max, far.k_max)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-6:
            continue  # no shared, data-supported region to compare

        grid = np.linspace(lo, hi, n)
        gap = svi_total_variance(grid, far.params) - svi_total_variance(grid, near.params)
        worst = float(np.min(gap))
        if worst < -1e-8:
            violations.append(
                {
                    "near_expiry": near.expiry,
                    "far_expiry": far.expiry,
                    "worst_gap": worst,
                    "k_at_worst": float(grid[int(np.argmin(gap))]),
                    "frac_k_violating": float(np.mean(gap < 0)),
                    "k_lo": float(lo),
                    "k_hi": float(hi),
                }
            )

    return len(violations) == 0, violations
