"""SVI model algebra, calibration recovery, and arbitrage diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.svi import (
    SVIFit,
    SVIParams,
    check_butterfly,
    check_calendar_arbitrage,
    durrleman_g,
    fit_slice,
    svi_derivatives,
    svi_implied_vol,
    svi_total_variance,
)

# A realistic equity-index slice: negative rho (downside skew), small a.
TRUE = SVIParams(a=0.012, b=0.115, rho=-0.62, m=0.021, sigma=0.135)
T_REF = 0.5


def _k_grid(n=61, lo=-0.45, hi=0.30):
    return np.linspace(lo, hi, n)


# ---------------------------------------------------------------------------
# Model algebra
# ---------------------------------------------------------------------------


def test_total_variance_formula():
    """Evaluate the closed form directly against the definition."""
    k = 0.1
    x = k - TRUE.m
    expected = TRUE.a + TRUE.b * (TRUE.rho * x + np.sqrt(x**2 + TRUE.sigma**2))
    assert float(svi_total_variance(k, TRUE)) == pytest.approx(expected, abs=1e-15)


def test_derivatives_match_finite_differences():
    k = _k_grid(21)
    h = 1e-5
    w1, w2 = svi_derivatives(k, TRUE)

    fd1 = (svi_total_variance(k + h, TRUE) - svi_total_variance(k - h, TRUE)) / (2 * h)
    fd2 = (
        svi_total_variance(k + h, TRUE)
        - 2 * svi_total_variance(k, TRUE)
        + svi_total_variance(k - h, TRUE)
    ) / h**2

    np.testing.assert_allclose(w1, fd1, atol=1e-8)
    np.testing.assert_allclose(w2, fd2, atol=1e-4)


def test_minimum_total_variance_is_attained_at_the_analytic_argmin():
    """w attains its minimum at k = m - rho*sigma/sqrt(1-rho^2)."""
    k_star = TRUE.m - TRUE.rho * TRUE.sigma / np.sqrt(1 - TRUE.rho**2)
    grid = np.linspace(k_star - 1.0, k_star + 1.0, 200_001)
    w = svi_total_variance(grid, TRUE)
    assert float(np.min(w)) == pytest.approx(TRUE.min_total_variance, abs=1e-9)
    assert float(grid[int(np.argmin(w))]) == pytest.approx(k_star, abs=1e-4)


def test_wing_slopes_are_the_asymptotic_slopes():
    left, right = TRUE.wing_slopes
    k = 1e7
    slope_right = (svi_total_variance(k + 1, TRUE) - svi_total_variance(k, TRUE)) / 1.0
    slope_left = (svi_total_variance(-k, TRUE) - svi_total_variance(-k - 1, TRUE)) / 1.0
    assert float(slope_right) == pytest.approx(right, rel=1e-6)
    assert float(-slope_left) == pytest.approx(left, rel=1e-6)


def test_implied_vol_is_sqrt_of_w_over_T():
    k = _k_grid(11)
    expected = np.sqrt(svi_total_variance(k, TRUE) / T_REF)
    np.testing.assert_allclose(svi_implied_vol(k, T_REF, TRUE), expected, atol=1e-15)


def test_negative_rho_produces_downside_skew():
    """The defining feature of an equity-index smile."""
    iv_down = float(svi_implied_vol(-0.15, T_REF, TRUE))
    iv_atm = float(svi_implied_vol(0.0, T_REF, TRUE))
    iv_up = float(svi_implied_vol(0.15, T_REF, TRUE))
    assert iv_down > iv_atm > iv_up


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_fit_recovers_parameters_on_noiseless_data():
    k = _k_grid(61)
    w = svi_total_variance(k, TRUE)

    fit = fit_slice(k, w, T=T_REF, expiry="synthetic", n_starts=24, seed=3)

    assert fit.rmse_total_var < 1e-8
    assert fit.params.a == pytest.approx(TRUE.a, abs=1e-4)
    assert fit.params.b == pytest.approx(TRUE.b, abs=1e-4)
    assert fit.params.rho == pytest.approx(TRUE.rho, abs=1e-3)
    assert fit.params.m == pytest.approx(TRUE.m, abs=1e-3)
    assert fit.params.sigma == pytest.approx(TRUE.sigma, abs=1e-3)


def test_fit_recovers_parameters_with_low_noise():
    """Low-noise recovery: 1e-5 of total variance, roughly a tick of vol."""
    rng = np.random.default_rng(11)
    k = _k_grid(81)
    w = svi_total_variance(k, TRUE) + rng.normal(0.0, 1e-5, size=k.size)

    fit = fit_slice(k, w, T=T_REF, expiry="synthetic_noisy", n_starts=32, seed=5)

    assert fit.params.a == pytest.approx(TRUE.a, abs=5e-3)
    assert fit.params.b == pytest.approx(TRUE.b, abs=1e-2)
    assert fit.params.rho == pytest.approx(TRUE.rho, abs=0.1)
    assert fit.params.m == pytest.approx(TRUE.m, abs=0.05)
    assert fit.params.sigma == pytest.approx(TRUE.sigma, abs=0.05)

    # More importantly: the fitted *curve* must track the true curve closely,
    # which is the thing the surface actually depends on.
    fine = _k_grid(401)
    err = svi_total_variance(fine, fit.params) - svi_total_variance(fine, TRUE)
    assert float(np.max(np.abs(err))) < 5e-5
    assert fit.rmse_iv < 5e-4  # under 5 vol bps


@pytest.mark.parametrize(
    "true",
    [
        SVIParams(a=0.004, b=0.06, rho=-0.80, m=0.005, sigma=0.06),  # short dated, sharp
        SVIParams(a=0.040, b=0.30, rho=-0.45, m=0.050, sigma=0.35),  # long dated, flat
        SVIParams(a=0.020, b=0.18, rho=0.10, m=-0.030, sigma=0.20),  # mild positive skew
    ],
)
def test_fit_recovers_a_range_of_slice_shapes(true):
    k = _k_grid(71, -0.6, 0.4)
    w = svi_total_variance(k, true)
    fit = fit_slice(k, w, T=1.0, n_starts=32, seed=17)

    fine = _k_grid(401, -0.6, 0.4)
    err = svi_total_variance(fine, fit.params) - svi_total_variance(fine, true)
    assert float(np.max(np.abs(err))) < 1e-6
    assert fit.rmse_total_var < 1e-7


def test_fit_reports_error_metrics_consistently():
    k = _k_grid(41)
    w = svi_total_variance(k, TRUE) + 1e-4
    fit = fit_slice(k, w, T=T_REF, n_starts=16, seed=2)

    model = svi_total_variance(k, fit.params)
    expected_rmse = float(np.sqrt(np.mean((model - w) ** 2)))
    assert fit.rmse_total_var == pytest.approx(expected_rmse, rel=1e-9)
    assert fit.max_abs_err_iv >= fit.rmse_iv
    assert 0.0 <= fit.r_squared <= 1.0
    assert fit.n_points == k.size


def test_fit_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 5 points"):
        fit_slice(np.array([0.0, 0.1, 0.2]), np.array([0.01, 0.011, 0.012]), T=1.0)


def test_fit_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        fit_slice(np.zeros(10), np.zeros(9), T=1.0)


def test_weights_shift_the_fit_toward_the_weighted_region():
    """Heavily weighting the left wing must reduce error there."""
    rng = np.random.default_rng(23)
    k = _k_grid(61)
    w = svi_total_variance(k, TRUE) + rng.normal(0, 5e-4, k.size)

    left = k < -0.2
    weights = np.where(left, 100.0, 1.0)

    flat_fit = fit_slice(k, w, T=T_REF, n_starts=16, seed=1)
    wtd_fit = fit_slice(k, w, T=T_REF, weights=weights, n_starts=16, seed=1)

    def left_err(fit):
        return float(np.sqrt(np.mean((svi_total_variance(k[left], fit.params) - w[left]) ** 2)))

    assert left_err(wtd_fit) < left_err(flat_fit)


def test_fitted_slice_is_a_valid_variance_curve():
    k = _k_grid(51)
    w = svi_total_variance(k, TRUE)
    fit = fit_slice(k, w, T=T_REF, n_starts=16, seed=4)
    assert fit.min_total_variance >= -1e-10
    assert np.all(svi_total_variance(_k_grid(501, -3, 3), fit.params) >= -1e-10)


# ---------------------------------------------------------------------------
# Arbitrage diagnostics
# ---------------------------------------------------------------------------


def test_durrleman_matches_its_definition():
    k = _k_grid(31)
    w = svi_total_variance(k, TRUE)
    w1, w2 = svi_derivatives(k, TRUE)
    expected = (1 - k * w1 / (2 * w)) ** 2 - (w1**2 / 4) * (1 / w + 0.25) + w2 / 2
    np.testing.assert_allclose(durrleman_g(k, TRUE), expected, atol=1e-14)


def test_well_behaved_slice_is_butterfly_arbitrage_free():
    ok, min_g = check_butterfly(TRUE, -1.0, 1.0)
    assert ok
    assert min_g > 0


def test_extreme_wing_slope_is_flagged_as_arbitrage():
    """b(1+|rho|) far above Lee's bound of 2 must produce a negative density."""
    bad = SVIParams(a=0.001, b=1.8, rho=-0.95, m=0.0, sigma=0.02)
    ok, min_g = check_butterfly(bad, -1.0, 1.0)
    assert not ok
    assert min_g < 0


def test_negative_variance_slice_is_rejected():
    bad = SVIParams(a=-0.5, b=0.1, rho=-0.5, m=0.0, sigma=0.1)
    assert bad.min_total_variance < 0
    ok, _ = check_butterfly(bad, -1.0, 1.0)
    assert not ok


def test_calibration_flags_butterfly_status():
    k = _k_grid(61)
    w = svi_total_variance(k, TRUE)
    fit = fit_slice(k, w, T=T_REF, n_starts=16, seed=6)
    assert fit.butterfly_ok
    assert fit.min_durrleman_g > 0


def _fit_from(params: SVIParams, T: float, expiry: str) -> SVIFit:
    k = _k_grid(41)
    w = svi_total_variance(k, params)
    return fit_slice(k, w, T=T, expiry=expiry, n_starts=12, seed=9)


def test_calendar_arbitrage_free_term_structure_passes():
    """Total variance rising with maturity at every k is arbitrage free."""
    near = SVIParams(a=0.008, b=0.08, rho=-0.6, m=0.02, sigma=0.10)
    far = SVIParams(a=0.030, b=0.16, rho=-0.6, m=0.03, sigma=0.20)
    fits = [_fit_from(near, 0.25, "near"), _fit_from(far, 1.0, "far")]

    ok, violations = check_calendar_arbitrage(fits)
    assert ok
    assert violations == []


def test_crossing_slices_are_flagged_as_calendar_arbitrage():
    """A longer expiry cheaper than a shorter one must be caught."""
    near = SVIParams(a=0.050, b=0.16, rho=-0.6, m=0.02, sigma=0.20)
    far = SVIParams(a=0.010, b=0.08, rho=-0.6, m=0.02, sigma=0.10)
    fits = [_fit_from(near, 0.25, "near"), _fit_from(far, 1.0, "far")]

    ok, violations = check_calendar_arbitrage(fits)
    assert not ok
    assert violations[0]["near_expiry"] == "near"
    assert violations[0]["far_expiry"] == "far"
    assert violations[0]["worst_gap"] < 0


def test_params_roundtrip_through_array():
    arr = TRUE.as_array()
    assert SVIParams.from_array(arr) == TRUE
    assert set(TRUE.as_dict()) == {"a", "b", "rho", "m", "sigma"}


def test_fit_row_is_serialisable():
    k = _k_grid(41)
    w = svi_total_variance(k, TRUE)
    row = fit_slice(k, w, T=T_REF, expiry="2026-01-16", n_starts=12, seed=8).to_row()

    assert row["expiry"] == "2026-01-16"
    assert row["n_points"] == 41
    assert row["rmse_iv_bps"] == pytest.approx(row["rmse_iv"] * 1e4)
    assert row["atm_iv"] == pytest.approx(float(svi_implied_vol(0.0, T_REF, TRUE)), abs=1e-4)
