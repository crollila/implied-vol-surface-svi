"""Surface construction: day counts, filters, forwards, and round-trip accuracy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from synthetic import ASOF, DIV_YIELD, RATE, SPOT

from volsurface.config import Config
from volsurface.surface import build_surface, longest_monotone_subset, year_fraction


def _cfg(**kw) -> Config:
    base = dict(ticker="SYN", rate=RATE, dividend_yield=DIV_YIELD, min_days=7, max_expiries=12)
    base.update(kw)
    return Config(**base)


# ---------------------------------------------------------------------------
# Day count
# ---------------------------------------------------------------------------


def test_year_fraction_counts_to_the_expiry_close():
    asof = datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc)
    assert year_fraction("2026-01-16", asof) == pytest.approx(1 / 365, abs=1e-12)
    assert year_fraction("2027-01-15", asof) == pytest.approx(365 / 365, abs=1e-12)


def test_year_fraction_is_negative_after_expiry():
    asof = datetime(2026, 3, 1, 20, 0, tzinfo=timezone.utc)
    assert year_fraction("2026-02-20", asof) < 0


def test_year_fraction_assumes_utc_for_naive_input():
    naive = datetime(2026, 1, 15, 20, 0)
    aware = datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc)
    assert year_fraction("2026-06-30", naive) == year_fraction("2026-06-30", aware)


# ---------------------------------------------------------------------------
# Monotone subset helper
# ---------------------------------------------------------------------------


def test_monotone_subset_keeps_everything_when_already_sorted():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(longest_monotone_subset(v, True), [0, 1, 2, 3])


def test_monotone_subset_drops_the_single_bad_point():
    """One stale print must cost one point, not the rest of the curve."""
    v = np.array([1.0, 2.0, 99.0, 3.0, 4.0, 5.0])
    keep = longest_monotone_subset(v, True)
    assert 2 not in keep
    assert len(keep) == 5


def test_monotone_subset_handles_decreasing():
    v = np.array([5.0, 4.0, 100.0, 3.0, 2.0])
    keep = longest_monotone_subset(v, False)
    assert 2 not in keep
    assert np.all(np.diff(v[keep]) <= 0)


def test_monotone_subset_allows_ties():
    v = np.array([1.0, 1.0, 1.0])
    assert len(longest_monotone_subset(v, True)) == 3


def test_monotone_subset_on_empty_input():
    assert longest_monotone_subset(np.array([]), True).size == 0


def test_monotone_subset_is_maximal():
    rng = np.random.default_rng(0)
    v = rng.normal(size=200).cumsum()
    v[50] += 100.0  # inject an outlier
    keep = longest_monotone_subset(v, True)
    assert np.all(np.diff(v[keep]) >= 0)


# ---------------------------------------------------------------------------
# End-to-end surface construction against a known surface
# ---------------------------------------------------------------------------


def test_recovers_the_generating_implied_vols(synthetic_chain):
    """The whole clean-and-invert path must return the vols we priced with."""
    surface = build_surface(synthetic_chain.snapshot, _cfg())

    assert len(surface.points) > 100
    assert surface.points["expiry"].nunique() == 5

    errs = []
    for _, row in surface.points.iterrows():
        truth = synthetic_chain.true_iv(row["expiry"], float(row["k"]))
        errs.append(abs(float(row["iv"]) - truth))

    assert max(errs) < 1e-6, f"worst IV error {max(errs):.2e}"


def test_recovers_the_forward_from_put_call_parity(synthetic_chain):
    surface = build_surface(synthetic_chain.snapshot, _cfg())

    assert set(surface.forwards["forward_source"]) == {"put_call_parity"}
    for _, row in surface.forwards.iterrows():
        expected = synthetic_chain.forwards[row["expiry"]]
        assert float(row["forward"]) == pytest.approx(expected, rel=1e-8)


def test_carry_forward_used_when_parity_is_disabled(synthetic_chain):
    surface = build_surface(synthetic_chain.snapshot, _cfg(imply_forward=False))
    assert set(surface.forwards["forward_source"]) == {"carry"}
    for _, row in surface.forwards.iterrows():
        expected = SPOT * np.exp((RATE - DIV_YIELD) * float(row["T"]))
        assert float(row["forward"]) == pytest.approx(expected, rel=1e-12)


def test_only_out_of_the_money_options_are_kept(synthetic_chain):
    surface = build_surface(synthetic_chain.snapshot, _cfg())
    pts = surface.points
    calls = pts[pts["option_type"] == "call"]
    puts = pts[pts["option_type"] == "put"]
    assert np.all(calls["strike"].to_numpy() >= calls["forward"].to_numpy())
    assert np.all(puts["strike"].to_numpy() < puts["forward"].to_numpy())


def test_total_variance_is_consistent_with_iv(synthetic_chain):
    surface = build_surface(synthetic_chain.snapshot, _cfg())
    pts = surface.points
    np.testing.assert_allclose(
        pts["total_var"].to_numpy(dtype=float),
        (pts["iv"] ** 2 * pts["T"]).to_numpy(dtype=float),
        rtol=1e-12,
    )


def test_log_moneyness_is_measured_against_the_forward(synthetic_chain):
    surface = build_surface(synthetic_chain.snapshot, _cfg())
    pts = surface.points
    np.testing.assert_allclose(
        pts["k"].to_numpy(dtype=float),
        np.log(pts["strike"].to_numpy(dtype=float) / pts["forward"].to_numpy(dtype=float)),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Price-source hierarchy
# ---------------------------------------------------------------------------


def test_live_quotes_are_preferred_over_last_trades(synthetic_chain):
    surface = build_surface(synthetic_chain.snapshot, _cfg())
    assert surface.price_source_mix.get("mid", 0) == len(surface.points)
    assert surface.price_source_mix.get("last", 0) == 0


def test_last_trade_fallback_carries_a_quoteless_chain(quoteless_chain):
    """With no two-sided markets the surface must still build, from last trades."""
    surface = build_surface(quoteless_chain.snapshot, _cfg())

    assert surface.price_source_mix.get("last", 0) == len(surface.points)
    assert len(surface.points) > 100

    worst = max(
        abs(float(r["iv"]) - quoteless_chain.true_iv(r["expiry"], float(r["k"])))
        for _, r in surface.points.iterrows()
    )
    assert worst < 1e-6


def test_stale_trades_are_still_used_when_they_are_the_newest_session(stale_chain):
    """Freshness is relative to the data, so a whole-chain lag is not fatal."""
    surface = build_surface(stale_chain.snapshot, _cfg())
    assert len(surface.points) > 100


def test_disabling_the_last_trade_fallback_rejects_a_quoteless_chain(quoteless_chain):
    with pytest.raises(RuntimeError):
        build_surface(quoteless_chain.snapshot, _cfg(allow_last_trade=False))


def test_a_single_stale_contract_is_dropped(chain_builder):
    """One contract left behind by the market must not reach the surface."""
    syn = chain_builder(two_sided=False)
    chain = syn.snapshot.chain
    victim = chain.index[len(chain) // 2]
    chain.loc[victim, "lastTradeDate"] = ASOF - timedelta(days=45)
    symbol = chain.loc[victim, "contractSymbol"]

    surface = build_surface(syn.snapshot, _cfg())
    assert symbol not in set(surface.points["contractSymbol"])


# ---------------------------------------------------------------------------
# Quote hygiene
# ---------------------------------------------------------------------------


def test_crossed_and_wide_markets_are_rejected(chain_builder):
    syn = chain_builder()
    chain = syn.snapshot.chain
    # Make every quote crossed and remove the last-trade fallback.
    chain["bid"] = chain["ask"] * 2.0
    chain["lastPrice"] = 0.0
    with pytest.raises(RuntimeError):
        build_surface(syn.snapshot, _cfg())


def test_sub_tick_premiums_are_dropped(synthetic_chain):
    surface = build_surface(synthetic_chain.snapshot, _cfg(min_price=0.5))
    assert np.all(surface.points["price"].to_numpy(dtype=float) >= 0.5)


def test_non_monotone_prices_are_filtered(chain_builder):
    """A price that breaks monotonicity in strike must not survive."""
    syn = chain_builder(two_sided=False)
    chain = syn.snapshot.chain
    expiry = sorted(syn.truth)[2]
    puts = chain[(chain["expiry"] == expiry) & (chain["option_type"] == "put")]
    victim = puts.sort_values("strike").index[len(puts) // 2]
    chain.loc[victim, "lastPrice"] *= 6.0  # wildly out of line with neighbours
    symbol = chain.loc[victim, "contractSymbol"]

    surface = build_surface(syn.snapshot, _cfg())
    assert symbol not in set(surface.points["contractSymbol"])


def test_thin_and_one_winged_slices_are_dropped(chain_builder):
    """A slice quoted only below the forward cannot identify rho and m."""
    syn = chain_builder()
    chain = syn.snapshot.chain
    expiry = sorted(syn.truth)[0]
    kill = (chain["expiry"] == expiry) & (chain["option_type"] == "call")
    chain.loc[kill, ["bid", "ask", "lastPrice"]] = 0.0

    surface = build_surface(syn.snapshot, _cfg())
    assert expiry not in surface.expiries


def test_expiry_window_is_respected(synthetic_chain):
    surface = build_surface(synthetic_chain.snapshot, _cfg(min_days=45, max_days=200))
    days = surface.forwards["days"].to_numpy(dtype=float)
    assert np.all(days >= 45) and np.all(days <= 200)


def test_max_expiries_spreads_across_the_term_structure(synthetic_chain):
    surface = build_surface(synthetic_chain.snapshot, _cfg(max_expiries=3))
    assert len(surface.expiries) == 3
    # Front and back must both survive, not just the three nearest.
    days = sorted(surface.forwards["days"].to_numpy(dtype=float))
    assert days[0] < 40 and days[-1] > 300


def test_sd_band_trims_more_aggressively_when_tightened(synthetic_chain):
    wide = build_surface(synthetic_chain.snapshot, _cfg(max_sd_moneyness=4.0))
    tight = build_surface(synthetic_chain.snapshot, _cfg(max_sd_moneyness=1.0))
    assert len(tight.points) < len(wide.points)

    for expiry in tight.expiries:
        slc = tight.slice(expiry)
        T = float(slc["T"].iloc[0])
        atm_iv = float(slc["iv"].iloc[np.argmin(np.abs(slc["k"].to_numpy()))])
        assert np.all(np.abs(slc["k"].to_numpy()) <= 1.0 * atm_iv * np.sqrt(T) * 1.2)


def test_filter_stats_are_recorded_and_monotone(synthetic_chain):
    surface = build_surface(synthetic_chain.snapshot, _cfg())
    stats = surface.filter_stats
    assert stats["raw_contracts"] == len(synthetic_chain.snapshot.chain)
    assert stats["after_thin_slices"] == len(surface.points)
    assert stats["after_iv_level"] >= stats["after_moneyness"] >= stats["after_vega"]


def test_slice_accessor_returns_sorted_points(synthetic_chain):
    surface = build_surface(synthetic_chain.snapshot, _cfg())
    slc = surface.slice(surface.expiries[0])
    assert np.all(np.diff(slc["k"].to_numpy(dtype=float)) > 0)


def test_empty_after_filtering_raises_a_useful_error(synthetic_chain):
    with pytest.raises(RuntimeError, match="no quotes survived|no expiry"):
        build_surface(synthetic_chain.snapshot, _cfg(min_days=5000, max_days=6000))
