"""End-to-end pipeline: fetch -> clean -> invert -> calibrate -> report."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import ChainSnapshot, fetch_chain
from .plotting import make_all_figures
from .reporting import write_analysis
from .surface import SurfaceData, build_surface
from .svi import SVIFit, check_calendar_arbitrage, fit_slice, svi_implied_vol

log = logging.getLogger(__name__)

__all__ = ["PipelineResult", "run_pipeline", "fit_all_slices"]


@dataclass
class PipelineResult:
    """Everything a run produces, for programmatic use or reporting."""

    snapshot: ChainSnapshot
    surface: SurfaceData
    fits: list[SVIFit]
    params_table: pd.DataFrame
    calendar_ok: bool
    calendar_violations: list[dict]
    figures: list[Path]
    outputs: dict[str, Path]

    @property
    def n_slices(self) -> int:
        return len(self.fits)

    def summary(self) -> str:
        rmse = self.params_table["rmse_iv_bps"]
        bad_fly = int((~self.params_table["butterfly_ok"]).sum())
        return (
            f"{self.surface.ticker} @ {self.surface.spot:,.2f} "
            f"({self.surface.asof:%Y-%m-%d %H:%M UTC})\n"
            f"  slices fitted     : {self.n_slices}\n"
            f"  surface points    : {len(self.surface.points)}\n"
            f"  RMSE (vol bps)    : median {rmse.median():.1f}, "
            f"mean {rmse.mean():.1f}, worst {rmse.max():.1f}\n"
            f"  butterfly-free    : {self.n_slices - bad_fly}/{self.n_slices} slices\n"
            f"  calendar-arb free : {'yes' if self.calendar_ok else f'no ({len(self.calendar_violations)} crossings)'}"
        )


def fit_all_slices(surface: SurfaceData, cfg: Config) -> list[SVIFit]:
    """Calibrate raw SVI to every expiry slice on the surface."""
    fits: list[SVIFit] = []

    for expiry in surface.expiries:
        slc = surface.slice(expiry)
        if len(slc) < max(5, cfg.min_points_per_slice):
            log.warning("skipping %s: only %d usable points", expiry, len(slc))
            continue

        T = float(slc["T"].iloc[0])
        weights = slc["vega"].to_numpy(dtype=float) if cfg.vega_weighted else None

        try:
            fit = fit_slice(
                slc["k"].to_numpy(dtype=float),
                slc["total_var"].to_numpy(dtype=float),
                T=T,
                expiry=expiry,
                weights=weights,
                n_starts=cfg.n_starts,
                seed=cfg.seed,
            )
        except (ValueError, RuntimeError) as exc:
            log.warning("skipping %s: calibration failed (%s)", expiry, exc)
            continue

        log.info(
            "%s  T=%.3f  n=%3d  RMSE=%5.1f bps  rho=%+.3f  butterfly_ok=%s",
            expiry, T, fit.n_points, fit.rmse_iv * 1e4, fit.params.rho, fit.butterfly_ok,
        )
        fits.append(fit)

    if not fits:
        raise RuntimeError("no expiry slice could be calibrated")

    return sorted(fits, key=lambda f: f.T)


def _params_table(fits: list[SVIFit]) -> pd.DataFrame:
    return pd.DataFrame([f.to_row() for f in fits]).sort_values("T").reset_index(drop=True)


def _points_table(surface: SurfaceData, fits: list[SVIFit]) -> pd.DataFrame:
    """Surface points annotated with the fitted value and residual."""
    by_expiry = {f.expiry: f for f in fits}
    pts = surface.points[surface.points["expiry"].isin(by_expiry)].copy()

    model = np.empty(len(pts))
    for i, (expiry, k) in enumerate(zip(pts["expiry"], pts["k"])):
        fit = by_expiry[expiry]
        model[i] = float(svi_implied_vol(k, fit.T, fit.params))

    pts["iv_svi"] = model
    pts["resid_iv_bps"] = (model - pts["iv"].to_numpy(dtype=float)) * 1e4

    cols = [
        "expiry", "T", "strike", "forward", "forward_source", "k", "option_type",
        "bid", "ask", "mid", "rel_spread", "volume", "openInterest",
        "iv", "iv_svi", "resid_iv_bps", "total_var", "vega", "iv_method",
    ]
    return pts[[c for c in cols if c in pts.columns]].reset_index(drop=True)


def run_pipeline(cfg: Config) -> PipelineResult:
    """Run the whole thing: data -> surface -> SVI -> figures -> report."""
    cfg.ensure_dirs()

    snapshot = fetch_chain(cfg)
    log.info(
        "%s: %d contracts across %d expiries (source: %s)",
        snapshot.ticker, snapshot.n_contracts, len(snapshot.expiries), snapshot.source,
    )

    surface = build_surface(snapshot, cfg)
    fits = fit_all_slices(surface, cfg)

    calendar_ok, violations = check_calendar_arbitrage(fits)
    if not calendar_ok:
        log.warning("calendar arbitrage detected in %d adjacent slice pairs", len(violations))

    params_table = _params_table(fits)
    points_table = _points_table(surface, fits)

    outputs: dict[str, Path] = {}

    params_path = cfg.outputs_dir / "svi_params.csv"
    params_table.to_csv(params_path, index=False)
    outputs["svi_params"] = params_path

    points_path = cfg.outputs_dir / "surface_points.csv"
    points_table.to_csv(points_path, index=False)
    outputs["surface_points"] = points_path

    fwd_path = cfg.outputs_dir / "forwards.csv"
    surface.forwards.to_csv(fwd_path, index=False)
    outputs["forwards"] = fwd_path

    figures = make_all_figures(surface, fits, cfg.figures_dir)

    result = PipelineResult(
        snapshot=snapshot,
        surface=surface,
        fits=fits,
        params_table=params_table,
        calendar_ok=calendar_ok,
        calendar_violations=violations,
        figures=figures,
        outputs=outputs,
    )

    analysis_path = write_analysis(result, cfg.root / "ANALYSIS.md")
    outputs["analysis"] = analysis_path

    return result
