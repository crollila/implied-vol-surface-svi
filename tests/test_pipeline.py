"""End-to-end pipeline, caching, reporting and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic import ASOF, DIV_YIELD, RATE, build_chain

from volsurface.cli import build_parser, main
from volsurface.config import Config
from volsurface.data import _save_cache, list_cached, load_cached
from volsurface.pipeline import run_pipeline
from volsurface.svi import svi_implied_vol


@pytest.fixture
def project(tmp_path: Path):
    """A project root with a synthetic chain already in the cache."""
    syn = build_chain()
    cfg = Config(
        ticker="SYN",
        rate=RATE,
        dividend_yield=DIV_YIELD,
        root=tmp_path,
        offline=True,
        min_days=7,
    )
    cfg.ensure_dirs()
    _save_cache(cfg, syn.snapshot)
    return cfg, syn


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------


def test_cache_round_trip_preserves_the_chain(project):
    cfg, syn = project
    loaded = load_cached(cfg, "SYN")

    assert loaded is not None
    assert loaded.spot == pytest.approx(syn.snapshot.spot)
    assert loaded.asof == syn.snapshot.asof
    assert len(loaded.chain) == len(syn.snapshot.chain)
    assert loaded.source == "cache"
    pd.testing.assert_frame_equal(
        loaded.chain.sort_values("contractSymbol").reset_index(drop=True),
        syn.snapshot.chain.sort_values("contractSymbol").reset_index(drop=True),
        check_dtype=False,
    )


def test_cache_writes_a_readable_metadata_sidecar(project):
    cfg, syn = project
    meta_path = cfg.cache_dir / "SYN" / ASOF.strftime("%Y-%m-%d") / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["ticker"] == "SYN"
    assert meta["spot"] == pytest.approx(syn.snapshot.spot)
    assert meta["n_contracts"] == len(syn.snapshot.chain)
    assert len(meta["expiries"]) == 5


def test_list_cached_reports_the_snapshot(project):
    cfg, _ = project
    assert list_cached(cfg, "SYN") == [ASOF.strftime("%Y-%m-%d")]
    assert list_cached(cfg, "NOPE") == []


def test_offline_without_a_cache_fails_clearly(tmp_path):
    cfg = Config(ticker="NOPE", root=tmp_path, offline=True)
    cfg.ensure_dirs()
    with pytest.raises(RuntimeError, match="no cached chain"):
        run_pipeline(cfg)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_pipeline_recovers_the_generating_surface(project):
    """The headline test: fit the surface we built the chain from."""
    cfg, syn = project
    result = run_pipeline(cfg)

    assert result.n_slices == 5
    assert result.calendar_ok
    assert all(f.butterfly_ok for f in result.fits)

    for fit in result.fits:
        # Compare curves, not parameters: raw SVI has a mild (rho, m)
        # degeneracy, so two parameter sets can describe the same smile.
        grid = np.linspace(fit.k_min, fit.k_max, 200)
        model = np.asarray(svi_implied_vol(grid, fit.T, fit.params), dtype=float)
        truth = np.array([syn.true_iv(fit.expiry, float(k)) for k in grid])
        assert float(np.max(np.abs(model - truth))) < 5e-4, fit.expiry
        assert fit.rmse_iv < 1e-4, fit.expiry


def test_pipeline_writes_every_output(project):
    cfg, _ = project
    result = run_pipeline(cfg)

    for path in result.figures:
        assert path.exists() and path.stat().st_size > 5_000
    assert {p.name for p in result.figures} == {
        "surface_3d.png",
        "smiles.png",
        "term_structure.png",
        "fit_diagnostics.png",
    }

    for name in ("svi_params", "surface_points", "forwards", "analysis"):
        assert result.outputs[name].exists()

    params = pd.read_csv(result.outputs["svi_params"])
    assert len(params) == 5
    for col in ("a", "b", "rho", "m", "sigma", "rmse_iv_bps", "butterfly_ok"):
        assert col in params.columns

    points = pd.read_csv(result.outputs["surface_points"])
    assert len(points) == len(result.surface.points)
    assert {"iv", "iv_svi", "resid_iv_bps", "k", "forward"} <= set(points.columns)


def test_reported_residuals_match_the_fit(project):
    cfg, _ = project
    result = run_pipeline(cfg)
    points = pd.read_csv(result.outputs["surface_points"])
    np.testing.assert_allclose(
        points["resid_iv_bps"].to_numpy(dtype=float),
        (points["iv_svi"] - points["iv"]).to_numpy(dtype=float) * 1e4,
        atol=1e-6,
    )


def test_params_table_rmse_matches_the_fit_objects(project):
    cfg, _ = project
    result = run_pipeline(cfg)
    by_expiry = {f.expiry: f for f in result.fits}
    for row in result.params_table.itertuples():
        assert row.rmse_iv_bps == pytest.approx(by_expiry[row.expiry].rmse_iv * 1e4)


def test_summary_is_human_readable(project):
    cfg, _ = project
    result = run_pipeline(cfg)
    text = result.summary()
    assert "slices fitted" in text
    assert "calendar-arb free : yes" in text
    assert "5/5 slices" in text


def test_analysis_document_is_written_with_real_numbers(project):
    cfg, _ = project
    result = run_pipeline(cfg)
    text = result.outputs["analysis"].read_text(encoding="utf-8")

    for heading in (
        "## 1. Snapshot",
        "## 2. Fitted SVI parameters",
        "## 3. The skew",
        "## 4. Term structure",
        "## 5. How far this is from Black-Scholes",
        "## 6. Arbitrage diagnostics",
        "## 8. Caveats",
    ):
        assert heading in text, heading

    assert "No butterfly arbitrage" in text
    assert "No calendar arbitrage" in text
    # Every fitted expiry must appear in the parameter table.
    for fit in result.fits:
        assert fit.expiry in text
    # No unrendered format placeholders left behind.
    assert "{" not in text.replace("{{", "").replace("}}", "")


def test_vega_weighting_is_configurable(project):
    cfg, _ = project
    weighted = run_pipeline(cfg)
    cfg.vega_weighted = False
    unweighted = run_pipeline(cfg)
    # Both should fit a noiseless surface almost perfectly; the point is that
    # the flag is plumbed through and does not crash.
    assert weighted.n_slices == unweighted.n_slices
    assert unweighted.params_table["rmse_iv_bps"].max() < 5.0


def test_pipeline_is_deterministic(project):
    cfg, _ = project
    a = run_pipeline(cfg).params_table
    b = run_pipeline(cfg).params_table
    pd.testing.assert_frame_equal(a, b)


def test_noisy_chain_still_fits_within_tolerance(tmp_path):
    """A realistic amount of quote noise must not derail the calibration."""
    syn = build_chain(noise_bps=25.0, seed=3)
    cfg = Config(
        ticker="SYN", rate=RATE, dividend_yield=DIV_YIELD, root=tmp_path,
        offline=True, min_days=7,
    )
    cfg.ensure_dirs()
    _save_cache(cfg, syn.snapshot)

    result = run_pipeline(cfg)
    assert result.n_slices == 5
    assert result.params_table["rmse_iv_bps"].median() < 60.0
    for fit in result.fits:
        grid = np.linspace(fit.k_min, fit.k_max, 100)
        model = np.asarray(svi_implied_vol(grid, fit.T, fit.params), dtype=float)
        truth = np.array([syn.true_iv(fit.expiry, float(k)) for k in grid])
        assert float(np.max(np.abs(model - truth))) < 0.01  # under 1 vol point


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parser_defaults_match_the_documented_invocation():
    args = build_parser().parse_args(["--ticker", "SPY", "--rate", "0.04"])
    assert args.ticker == "SPY"
    assert args.rate == 0.04
    assert args.offline is False
    assert args.max_expiries == 12


def test_parser_flags_toggle_the_right_switches():
    args = build_parser().parse_args(
        ["--offline", "--no-vega-weight", "--no-last-trade", "--no-imply-forward"]
    )
    assert args.offline and args.no_vega_weight and args.no_last_trade
    assert args.no_imply_forward


def test_cli_runs_end_to_end_offline(project, monkeypatch, capsys):
    cfg, _ = project
    monkeypatch.chdir(cfg.root)

    code = main(["--ticker", "SYN", "--rate", str(RATE), "--offline", "-q"])
    assert code == 0

    out = capsys.readouterr().out
    assert "slices fitted" in out
    assert (cfg.root / "figures" / "surface_3d.png").exists()
    assert (cfg.root / "outputs" / "svi_params.csv").exists()
    assert (cfg.root / "ANALYSIS.md").exists()


def test_cli_reports_failure_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["--ticker", "NOPE", "--offline", "-q"])
    assert code == 1
    assert "volsurface:" in capsys.readouterr().err
