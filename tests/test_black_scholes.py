"""Black-Scholes pricing and greeks: correctness against known values."""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.black_scholes import (
    bs_call_price,
    bs_price,
    bs_put_price,
    bs_vega,
    d1_d2,
    forward_price,
    intrinsic_value,
    norm_cdf,
    price_bounds,
)

# Reference prices computed independently with mpmath at 50 decimal places
# (see the note in tests/README or the project history).  ``hull_15_6`` and
# ``atm_1y`` are the standard textbook cases -- Hull's Example 15.6 quotes
# 4.76 / 0.81, and the 100/100/1y/5%/20% call is the widely published
# 10.4506.
CASES = [
    dict(label="hull_15_6", S=42, K=40, T=0.5, r=0.1, q=0.0, sigma=0.2,
         call=4.7594223928715334, put=0.80859937290009365, vega=8.8134150596028514),
    dict(label="atm_1y", S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.2,
         call=10.450583572185567, put=5.573526022256968, vega=37.524034691693788),
    dict(label="otm_short", S=100, K=120, T=0.25, r=0.03, q=0.0, sigma=0.35,
         call=1.5904281497022134, put=20.693794727998825, vega=13.166638990040832),
    dict(label="itm_div", S=100, K=90, T=2.0, r=0.04, q=0.02, sigma=0.25,
         call=20.103221276069814, put=7.1047485356347135, vega=45.60335403366368),
    dict(label="low_vol", S=50, K=55, T=0.75, r=0.01, q=0.01, sigma=0.1,
         call=0.30878118974587554, put=5.2714214638415677, vega=9.8046719428947102),
    dict(label="high_vol", S=250, K=200, T=1.5, r=0.045, q=0.015, sigma=0.8,
         call=112.56397456943933, put=55.07170939431075, vega=89.231556036152242),
]

IDS = [c["label"] for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_call_price_matches_reference(case):
    got = bs_call_price(case["S"], case["K"], case["T"], case["r"], case["q"], case["sigma"])
    assert float(got) == pytest.approx(case["call"], abs=1e-6)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_put_price_matches_reference(case):
    got = bs_put_price(case["S"], case["K"], case["T"], case["r"], case["q"], case["sigma"])
    assert float(got) == pytest.approx(case["put"], abs=1e-6)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_vega_matches_reference(case):
    got = bs_vega(case["S"], case["K"], case["T"], case["r"], case["q"], case["sigma"])
    assert float(got) == pytest.approx(case["vega"], abs=1e-6)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_put_call_parity(case):
    """C - P = S*exp(-qT) - K*exp(-rT) must hold exactly, to machine precision."""
    S, K, T, r, q, sigma = (case[x] for x in ("S", "K", "T", "r", "q", "sigma"))
    c = float(bs_call_price(S, K, T, r, q, sigma))
    p = float(bs_put_price(S, K, T, r, q, sigma))
    expected = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert (c - p) == pytest.approx(expected, abs=1e-10)


def test_put_call_parity_on_synthetic_grid():
    """Parity across a wide synthetic grid, not just the reference cases."""
    rng = np.random.default_rng(0)
    n = 500
    S = rng.uniform(20, 500, n)
    K = S * np.exp(rng.uniform(-0.8, 0.8, n))
    T = rng.uniform(0.01, 3.0, n)
    sigma = rng.uniform(0.05, 1.2, n)
    r, q = 0.037, 0.014

    c = bs_call_price(S, K, T, r, q, sigma)
    p = bs_put_price(S, K, T, r, q, sigma)
    expected = S * np.exp(-q * T) - K * np.exp(-r * T)

    np.testing.assert_allclose(c - p, expected, atol=1e-9, rtol=0.0)


def test_vega_matches_finite_difference():
    """Analytic vega agrees with a central difference of the price."""
    S, K, T, r, q, sigma = 105.0, 98.0, 0.8, 0.03, 0.01, 0.27
    h = 1e-6
    up = float(bs_call_price(S, K, T, r, q, sigma + h))
    dn = float(bs_call_price(S, K, T, r, q, sigma - h))
    fd = (up - dn) / (2 * h)
    assert float(bs_vega(S, K, T, r, q, sigma)) == pytest.approx(fd, rel=1e-6)


def test_vega_identical_for_calls_and_puts():
    S, K, T, r, q, sigma = 100.0, 110.0, 1.3, 0.02, 0.01, 0.3
    h = 1e-6
    fd_put = (
        float(bs_put_price(S, K, T, r, q, sigma + h))
        - float(bs_put_price(S, K, T, r, q, sigma - h))
    ) / (2 * h)
    assert float(bs_vega(S, K, T, r, q, sigma)) == pytest.approx(fd_put, rel=1e-6)


def test_monotone_in_volatility():
    """Price is strictly increasing in sigma -- the property the IV solver relies on."""
    sigmas = np.linspace(0.01, 2.0, 200)
    calls = bs_call_price(100.0, 95.0, 1.0, 0.03, 0.0, sigmas)
    puts = bs_put_price(100.0, 95.0, 1.0, 0.03, 0.0, sigmas)
    assert np.all(np.diff(calls) > 0)
    assert np.all(np.diff(puts) > 0)


def test_zero_vol_limit_is_discounted_intrinsic():
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.05, 0.01
    for kind in ("call", "put"):
        got = float(bs_price(S, K, T, r, q, 0.0, kind))
        assert got == pytest.approx(float(intrinsic_value(S, K, T, r, q, kind)), abs=1e-12)


def test_expired_option_is_intrinsic():
    assert float(bs_call_price(110.0, 100.0, 0.0, 0.05, 0.0, 0.3)) == pytest.approx(10.0)
    assert float(bs_put_price(90.0, 100.0, 0.0, 0.05, 0.0, 0.3)) == pytest.approx(10.0)
    assert float(bs_call_price(90.0, 100.0, 0.0, 0.05, 0.0, 0.3)) == pytest.approx(0.0)


def test_prices_respect_no_arbitrage_bounds():
    rng = np.random.default_rng(1)
    S = rng.uniform(20, 300, 300)
    K = S * np.exp(rng.uniform(-1.0, 1.0, 300))
    T = rng.uniform(0.02, 2.5, 300)
    sigma = rng.uniform(0.05, 1.5, 300)
    r, q = 0.04, 0.012

    for kind, fn in (("call", bs_call_price), ("put", bs_put_price)):
        price = fn(S, K, T, r, q, sigma)
        lower, upper = price_bounds(S, K, T, r, q, kind)
        assert np.all(price >= lower - 1e-9)
        assert np.all(price <= upper + 1e-9)


def test_norm_cdf_known_values():
    assert float(norm_cdf(0.0)) == pytest.approx(0.5, abs=1e-15)
    assert float(norm_cdf(1.96)) == pytest.approx(0.9750021048517795, abs=1e-12)
    assert float(norm_cdf(-1.0)) == pytest.approx(0.15865525393145707, abs=1e-12)


def test_d1_d2_relationship():
    S, K, T, r, q, sigma = 100.0, 105.0, 0.6, 0.03, 0.01, 0.22
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    assert float(d1 - d2) == pytest.approx(sigma * np.sqrt(T), abs=1e-14)


def test_forward_price():
    assert float(forward_price(100.0, 2.0, 0.05, 0.02)) == pytest.approx(
        100.0 * np.exp(0.06), abs=1e-12
    )


def test_vectorisation_matches_scalar_loop():
    S = np.array([90.0, 100.0, 110.0])
    K = np.array([100.0, 100.0, 100.0])
    T = np.array([0.5, 1.0, 1.5])
    sigma = np.array([0.2, 0.25, 0.3])
    vec = bs_call_price(S, K, T, 0.03, 0.01, sigma)
    loop = [float(bs_call_price(s, k, t, 0.03, 0.01, v)) for s, k, t, v in zip(S, K, T, sigma)]
    np.testing.assert_allclose(vec, loop, atol=1e-14)
