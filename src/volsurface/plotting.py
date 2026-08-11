"""Figure generation for the fitted volatility surface.

All plots are written to ``figures/`` as PNG.  Matplotlib runs on the Agg
backend so the pipeline works headless (CI, WSL, a container) without a
display.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm  # noqa: E402

from .surface import SurfaceData  # noqa: E402
from .svi import SVIFit, svi_implied_vol, svi_total_variance  # noqa: E402

log = logging.getLogger(__name__)

__all__ = [
    "plot_surface_3d",
    "plot_smiles",
    "plot_term_structure",
    "plot_fit_diagnostics",
    "make_all_figures",
]

_DPI = 150
_MARKET_STYLE = dict(s=18, color="#d1495b", alpha=0.85, zorder=3, edgecolors="none")
_FIT_STYLE = dict(color="#1b3b6f", lw=2.0, zorder=2)


def _title(surface: SurfaceData, what: str) -> str:
    return (
        f"{surface.ticker} {what}\n"
        f"spot {surface.spot:,.2f} | {surface.asof:%Y-%m-%d %H:%M UTC} | "
        f"r = {surface.rate:.2%}"
    )


def _atm_vol(fit: SVIFit) -> float:
    return float(svi_implied_vol(0.0, fit.T, fit.params))


def plot_surface_3d(surface: SurfaceData, fits: list[SVIFit], path: Path) -> Path:
    """The headline surface figure, in two panels.

    **Left — 3D surface in standardised moneyness.** Plotting against raw
    ``k = log(K/F)`` makes a ragged, unreadable mesh, because the quoted
    strike width grows with maturity: a one-week slice spans ``|k| < 0.06``
    while an 18-month slice spans ``|k| < 0.8``, so any common ``k`` axis is
    either a sliver or mostly empty. Rescaling to

    .. math:: z = \\frac{k}{\\sigma_{ATM}\\sqrt{T}}

    -- log-moneyness in standard deviations of return to expiry -- puts every
    slice on a comparable footing and yields a clean rectangular mesh. It is
    also the more meaningful comparison: ``z`` measures how far out of the
    money an option is *relative to what could plausibly happen* by expiry.

    **Right — the same surface in raw log-moneyness**, drawn as a fan of
    smiles, one line per expiry over exactly the strikes that expiry quotes.
    This preserves the true strike coverage that the left panel normalises
    away.
    """
    fits = sorted(fits, key=lambda f: f.T)

    fig = plt.figure(figsize=(17, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.12)

    # ---------------- Left: 3D surface in standardised moneyness ----------
    ax = fig.add_subplot(gs[0, 0], projection="3d")

    scales = {f.expiry: max(_atm_vol(f) * np.sqrt(f.T), 1e-9) for f in fits}
    z_lo = max(f.k_min / scales[f.expiry] for f in fits)
    z_hi = min(f.k_max / scales[f.expiry] for f in fits)

    z_grid = np.linspace(z_lo, z_hi, 140)
    t_grid = np.array([f.T for f in fits])
    Z_mesh, T_mesh = np.meshgrid(z_grid, t_grid)
    iv_mesh = np.vstack(
        [
            np.asarray(svi_implied_vol(z_grid * scales[f.expiry], f.T, f.params), float) * 100.0
            for f in fits
        ]
    )

    surf = ax.plot_surface(
        Z_mesh, T_mesh, iv_mesh,
        cmap=cm.viridis, alpha=0.9, linewidth=0.15, edgecolor="white",
        rstride=1, cstride=4, antialiased=True,
    )

    pts = surface.points.copy()
    pts["z"] = pts["k"] / pts["expiry"].map(scales)
    pts = pts[(pts["z"] >= z_lo) & (pts["z"] <= z_hi)]
    ax.scatter(
        pts["z"], pts["T"], pts["iv"] * 100.0,
        s=6, c="#d1495b", depthshade=False, label="market IV", alpha=0.6,
    )

    ax.set_xlabel("standardised moneyness  z = k / (σ√T)", labelpad=12)
    ax.set_ylabel("time to expiry (years)", labelpad=12)
    ax.set_zlabel("implied volatility (%)", labelpad=8)
    ax.set_title("Fitted SVI surface", pad=14)
    ax.view_init(elev=24, azim=-128)
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=16, pad=0.09, label="IV (%)")

    # ---------------- Right: fan of smiles in raw log-moneyness -----------
    ax2 = fig.add_subplot(gs[0, 1])
    cmap = cm.viridis
    t_min, t_max = float(t_grid.min()), float(t_grid.max())

    for f in fits:
        frac = (f.T - t_min) / (t_max - t_min) if t_max > t_min else 0.5
        colour = cmap(frac)
        grid = np.linspace(f.k_min, f.k_max, 240)
        ax2.plot(
            grid, np.asarray(svi_implied_vol(grid, f.T, f.params), float) * 100.0,
            color=colour, lw=1.8, label=f"{f.T * 365:.0f}d",
        )
        slc = surface.slice(f.expiry)
        ax2.scatter(slc["k"], slc["iv"] * 100.0, s=5, color=colour, alpha=0.35)

    ax2.axvline(0.0, color="grey", lw=0.8, ls=":")
    ax2.set_xlabel("log-moneyness  k = log(K/F)")
    ax2.set_ylabel("implied volatility (%)")
    ax2.set_title("Smiles by expiry, over their quoted strike range")
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.legend(fontsize=7, ncol=2, frameon=False, title="maturity", title_fontsize=7)

    fig.suptitle(_title(surface, "implied volatility surface — SVI fit"), y=1.02)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def plot_smiles(surface: SurfaceData, fits: list[SVIFit], path: Path) -> Path:
    """Per-expiry smiles: market IV points against the calibrated SVI curve."""
    fits = sorted(fits, key=lambda f: f.T)
    n = len(fits)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False, sharey=False
    )

    for idx, fit in enumerate(fits):
        ax = axes[idx // ncols][idx % ncols]
        slc = surface.slice(fit.expiry)

        pad = 0.03 * max(fit.k_max - fit.k_min, 0.05)
        grid = np.linspace(fit.k_min - pad, fit.k_max + pad, 300)
        model_iv = svi_implied_vol(grid, fit.T, fit.params) * 100.0

        ax.scatter(slc["k"], slc["iv"] * 100.0, label="market", **_MARKET_STYLE)
        ax.plot(grid, model_iv, label="SVI fit", **_FIT_STYLE)
        ax.axvline(0.0, color="grey", lw=0.7, ls=":", zorder=1)

        flag = "" if fit.butterfly_ok else "  ⚠ butterfly"
        ax.set_title(
            f"{fit.expiry}  ({fit.T * 365:.0f}d, n={fit.n_points})\n"
            f"RMSE {fit.rmse_iv * 1e4:.0f} bps | ρ={fit.params.rho:+.2f}{flag}",
            fontsize=9,
        )
        ax.set_xlabel("k = log(K/F)", fontsize=8)
        ax.set_ylabel("IV (%)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.6)
        if idx == 0:
            ax.legend(fontsize=8, frameon=False)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(_title(surface, "volatility smiles by expiry"), y=1.005)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def plot_term_structure(surface: SurfaceData, fits: list[SVIFit], path: Path) -> Path:
    """ATM vol, skew, curvature and total variance across maturities."""
    fits = sorted(fits, key=lambda f: f.T)
    T = np.array([f.T for f in fits])
    days = T * 365.0

    atm_iv = np.array([float(svi_implied_vol(0.0, f.T, f.params)) for f in fits]) * 100.0

    # 25-delta-ish proxy skew: the IV difference across a +/-10% moneyness
    # band, which is how desks eyeball equity skew without a delta solve.
    band = 0.10
    skew = np.array(
        [
            float(svi_implied_vol(-band, f.T, f.params) - svi_implied_vol(band, f.T, f.params))
            for f in fits
        ]
    ) * 100.0

    atm_total_var = np.array([float(svi_total_variance(0.0, f.params)) for f in fits])
    rmse_bps = np.array([f.rmse_iv * 1e4 for f in fits])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0][0]
    ax.plot(days, atm_iv, "o-", color="#1b3b6f")
    ax.set_title("ATM implied volatility term structure")
    ax.set_xlabel("days to expiry")
    ax.set_ylabel("ATM IV (%)")

    ax = axes[0][1]
    ax.plot(days, skew, "o-", color="#d1495b")
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_title(f"Skew:  IV(k=−{band:.2f}) − IV(k=+{band:.2f})")
    ax.set_xlabel("days to expiry")
    ax.set_ylabel("skew (vol points)")

    ax = axes[1][0]
    ax.plot(days, atm_total_var, "o-", color="#2a9d8f")
    ax.set_title("ATM total variance  w = σ²T  (must be increasing)")
    ax.set_xlabel("days to expiry")
    ax.set_ylabel("total variance")

    ax = axes[1][1]
    ax.bar(range(len(fits)), rmse_bps, color="#6a4c93")
    ax.set_xticks(range(len(fits)))
    ax.set_xticklabels([f.expiry for f in fits], rotation=70, fontsize=7)
    ax.set_title("SVI fit error by slice")
    ax.set_ylabel("RMSE (vol bps)")

    for row in axes:
        for a in row:
            a.grid(alpha=0.25, lw=0.6)

    fig.suptitle(_title(surface, "term structure & fit quality"), y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def plot_fit_diagnostics(surface: SurfaceData, fits: list[SVIFit], path: Path) -> Path:
    """Residual structure and the arbitrage diagnostics behind the fit."""
    fits = sorted(fits, key=lambda f: f.T)

    resid_k: list[np.ndarray] = []
    resid_v: list[np.ndarray] = []
    for fit in fits:
        slc = surface.slice(fit.expiry)
        k = slc["k"].to_numpy(dtype=float)
        model = svi_implied_vol(k, fit.T, fit.params)
        resid_k.append(k)
        resid_v.append((model - slc["iv"].to_numpy(dtype=float)) * 1e4)

    all_k = np.concatenate(resid_k)
    all_r = np.concatenate(resid_v)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for fit, k, r in zip(fits, resid_k, resid_v):
        ax.scatter(k, r, s=10, alpha=0.6, label=fit.expiry)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Fit residuals vs log-moneyness")
    ax.set_xlabel("k = log(K/F)")
    ax.set_ylabel("model − market (vol bps)")
    if len(fits) <= 8:
        ax.legend(fontsize=6, ncol=2, frameon=False)

    ax = axes[1]
    ax.hist(all_r, bins=40, color="#1b3b6f", alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(
        f"Residual distribution\nmean {all_r.mean():+.1f} bps, sd {all_r.std():.1f} bps"
    )
    ax.set_xlabel("model − market (vol bps)")
    ax.set_ylabel("count")

    ax = axes[2]
    from .svi import durrleman_g

    for fit in fits:
        grid = np.linspace(fit.k_min, fit.k_max, 200)
        ax.plot(grid, durrleman_g(grid, fit.params), lw=1.2, alpha=0.8)
    ax.axhline(0, color="red", lw=1.0, ls="--")
    ax.set_title("Durrleman g(k) — must stay above zero\n(butterfly-arbitrage check)")
    ax.set_xlabel("k = log(K/F)")
    ax.set_ylabel("g(k)")

    for a in axes:
        a.grid(alpha=0.25, lw=0.6)

    fig.suptitle(_title(surface, "calibration diagnostics"), y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s (n=%d residuals)", path, all_k.size)
    return path


def make_all_figures(surface: SurfaceData, fits: list[SVIFit], figures_dir: Path) -> list[Path]:
    """Generate the full figure set; returns the paths written."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_surface_3d(surface, fits, figures_dir / "surface_3d.png"),
        plot_smiles(surface, fits, figures_dir / "smiles.png"),
        plot_term_structure(surface, fits, figures_dir / "term_structure.png"),
        plot_fit_diagnostics(surface, fits, figures_dir / "fit_diagnostics.png"),
    ]
