"""Implied-volatility solver: round-trip accuracy and graceful failure."""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.black_scholes import bs_call_price, bs_price, bs_put_price
from volsurface.implied_vol import implied_vol, implied_vol_vectorised

ROUND_TRIP_TOL = 1e-4


@pytest.mark.parametrize("sigma", [0.05, 0.10, 0.20, 0.35, 0.60, 1.00, 1.50])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_round_trip_atm(sigma, kind):
    """Price at a known vol, recover it to within 1e-4."""
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    price = float(bs_price(S, K, T, r, q, sigma, kind))
    res = implied_vol(price, S, K, T, r, q, kind)
    assert res.ok, res.reason
    assert res.iv == pytest.approx(sigma, abs=ROUND_TRIP_TOL)


def test_round_trip_across_a_wide_grid():
    """Round-trip over a realistic grid of strikes, maturities and vols.

    Only quotes with meaningful time value are asserted on; a deep in-the-
    money option whose time value underflows carries no vol information and
    the solver is expected to refuse it (checked separately below).
    """
    rng = np.random.default_rng(42)
    r, q = 0.04, 0.012
    S = 100.0

    n_checked = 0
    for _ in range(400):
        T = float(rng.uniform(0.02, 2.5))
        sigma = float(rng.uniform(0.05, 1.2))
        k = float(rng.uniform(-1.0, 1.0))
        K = S * np.exp(k)
        kind = "call" if k >= 0 else "put"  # OTM side, as the pipeline uses

        price = float(bs_price(S, K, T, r, q, sigma, kind))
        res = implied_vol(price, S, K, T, r, q, kind)

        if not res.ok:
            # Acceptable only when the quote genuinely holds no information:
            # a far-OTM premium that underflows to zero, or an ITM price
            # sitting on discounted intrinsic.
            assert res.reason in {
                "zero_time_value",
                "below_intrinsic",
                "non_positive_price",
            }, res.reason
            continue

        n_checked += 1
        assert res.iv == pytest.approx(sigma, abs=ROUND_TRIP_TOL), (
            f"T={T} K={K} sigma={sigma} kind={kind} got={res.iv}"
        )

    assert n_checked > 350, f"too many quotes rejected: only {n_checked} checked"


@pytest.mark.parametrize("k", [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9])
def test_round_trip_itm_and_otm_both_sides(k):
    """Both calls and puts recover the vol, including in the money."""
    S, T, r, q, sigma = 100.0, 0.75, 0.03, 0.015, 0.28
    K = S * np.exp(k)
    for kind in ("call", "put"):
        price = float(bs_price(S, K, T, r, q, sigma, kind))
        res = implied_vol(price, S, K, T, r, q, kind)
        assert res.ok, f"{kind} k={k}: {res.reason}"
        assert res.iv == pytest.approx(sigma, abs=ROUND_TRIP_TOL)


def test_solver_reports_which_method_was_used():
    price = float(bs_call_price(100.0, 100.0, 1.0, 0.03, 0.0, 0.2))
    res = implied_vol(price, 100.0, 100.0, 1.0, 0.03, 0.0, "call")
    assert res.method in {"brent", "newton"}
    assert res.reason == ""


def test_price_below_intrinsic_is_rejected():
    S, K, T, r, q = 100.0, 80.0, 1.0, 0.05, 0.0
    intrinsic = S - K * np.exp(-r * T)
    res = implied_vol(intrinsic * 0.5, S, K, T, r, q, "call")
    assert not res.ok
    assert res.reason == "below_intrinsic"


def test_price_above_upper_bound_is_rejected():
    """A call cannot be worth more than the discounted spot."""
    S, K, T, r, q = 100.0, 80.0, 1.0, 0.05, 0.0
    res = implied_vol(S * 1.01, S, K, T, r, q, "call")
    assert not res.ok
    assert res.reason == "above_upper_bound"


@pytest.mark.parametrize(
    "price,S,K,T,expected",
    [
        (5.0, 100.0, 100.0, 0.0, "expired"),
        (5.0, 100.0, 100.0, -0.1, "expired"),
        (0.0, 100.0, 100.0, 1.0, "non_positive_price"),
        (-1.0, 100.0, 100.0, 1.0, "non_positive_price"),
        (np.nan, 100.0, 100.0, 1.0, "non_finite_input"),
        (5.0, 0.0, 100.0, 1.0, "non_positive_spot_or_strike"),
    ],
)
def test_degenerate_inputs_fail_gracefully(price, S, K, T, expected):
    res = implied_vol(price, S, K, T, 0.03, 0.0, "call")
    assert not res.ok
    assert res.reason == expected
    assert np.isnan(res.iv)


def test_deep_itm_call_with_no_time_value_is_rejected_not_faked():
    """A quote pinned at intrinsic must be refused rather than given a fake IV."""
    S, K, T, r, q = 100.0, 10.0, 0.05, 0.04, 0.0
    intrinsic = S - K * np.exp(-r * T)
    res = implied_vol(intrinsic, S, K, T, r, q, "call")
    assert not res.ok
    assert np.isnan(res.iv)


def test_vectorised_matches_scalar():
    S, T, r, q = 100.0, 0.5, 0.03, 0.01
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    sigmas = np.array([0.35, 0.30, 0.26, 0.28, 0.32])
    kinds = np.array(["put", "put", "call", "call", "call"], dtype=object)

    prices = np.array(
        [float(bs_price(S, k, T, r, q, s, t)) for k, s, t in zip(strikes, sigmas, kinds)]
    )
    iv, method, reason = implied_vol_vectorised(prices, S, strikes, T, r, q, kinds)

    np.testing.assert_allclose(iv, sigmas, atol=ROUND_TRIP_TOL)
    assert all(m in {"brent", "newton"} for m in method)
    assert all(r_ == "" for r_ in reason)


def test_vectorised_isolates_bad_rows():
    """One unusable quote must not contaminate its neighbours."""
    S, T, r, q = 100.0, 0.5, 0.03, 0.0
    strikes = np.array([95.0, 100.0, 105.0])
    good_call = float(bs_call_price(S, 105.0, T, r, q, 0.25))
    good_put = float(bs_put_price(S, 95.0, T, r, q, 0.30))
    prices = np.array([good_put, -1.0, good_call])  # middle quote is garbage
    kinds = np.array(["put", "call", "call"], dtype=object)

    iv, _, reason = implied_vol_vectorised(prices, S, strikes, T, r, q, kinds)

    assert iv[0] == pytest.approx(0.30, abs=ROUND_TRIP_TOL)
    assert np.isnan(iv[1])
    assert reason[1] == "non_positive_price"
    assert iv[2] == pytest.approx(0.25, abs=ROUND_TRIP_TOL)


def test_recovers_very_short_dated_vol():
    """One-day options are where naive solvers break down."""
    S, K, T, r, q, sigma = 100.0, 101.0, 1.0 / 365.0, 0.04, 0.0, 0.18
    price = float(bs_call_price(S, K, T, r, q, sigma))
    res = implied_vol(price, S, K, T, r, q, "call")
    assert res.ok, res.reason
    assert res.iv == pytest.approx(sigma, abs=ROUND_TRIP_TOL)


def test_round_trip_precision_is_much_better_than_tolerance():
    """The solver should be near machine precision, not merely inside 1e-4."""
    S, K, T, r, q, sigma = 100.0, 105.0, 1.0, 0.03, 0.01, 0.2237
    price = float(bs_call_price(S, K, T, r, q, sigma))
    res = implied_vol(price, S, K, T, r, q, "call")
    assert res.iv == pytest.approx(sigma, abs=1e-10)
