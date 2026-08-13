"""Explanatory figures for the README, for readers new to the terminology.

These are *derived from the package's own model code* rather than drawn by
hand: every curve here is produced by the same Black-Scholes and SVI
functions the pipeline uses, so the illustrations cannot drift away from
what the code actually does.

Run with::

    python -m volsurface.explainers
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from .black_scholes import bs_call_price, bs_put_price, bs_vega  # noqa: E402
from .svi import SVIParams, svi_implied_vol, svi_total_variance  # noqa: E402

log = logging.getLogger(__name__)

__all__ = ["make_all_explainers", "main"]

_DPI = 150

# A representative equity-index slice, used across the explainers.
_REF = SVIParams(a=0.012, b=0.115, rho=-0.62, m=0.021, sigma=0.135)
_T = 0.5

_INK = "#1b3b6f"
_ACCENT = "#d1495b"
_GREY = "#8a8f98"
_GREEN = "#2a9d8f"


def _style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=11, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.22, lw=0.6)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# 1. What an implied volatility even is
# ---------------------------------------------------------------------------


def explain_implied_vol(path: Path) -> Path:
    """Show that pricing is a one-way function, and IV is the inverse.

    The left panel is the honest statement of the problem: option price is a
    strictly increasing function of volatility, so a quoted price pins down
    exactly one volatility. The right panel shows why that inversion needs a
    real root finder rather than a formula -- vega, the slope being inverted,
    collapses toward zero in the wings.
    """
    S, K, T, r = 100.0, 100.0, 1.0, 0.04
    sigmas = np.linspace(0.01, 1.0, 400)
    prices = np.asarray(bs_call_price(S, K, T, r, 0.0, sigmas), dtype=float)

    market_price = float(bs_call_price(S, K, T, r, 0.0, 0.25))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ax.plot(sigmas * 100, prices, color=_INK, lw=2.2)
    ax.axhline(market_price, color=_ACCENT, lw=1.4, ls="--")
    ax.axvline(25.0, color=_ACCENT, lw=1.4, ls="--")
    ax.plot([25.0], [market_price], "o", color=_ACCENT, ms=9, zorder=5)

    ax.annotate(
        f"① the market quotes\nthis option at ${market_price:.2f}",
        xy=(11, market_price), xytext=(2, market_price + 8),
        fontsize=9, color=_ACCENT,
        arrowprops=dict(arrowstyle="->", color=_ACCENT, lw=1.2),
    )
    ax.annotate(
        "② so its implied\nvolatility is 25%",
        xy=(25.0, market_price + 1.5), xytext=(31, market_price + 14),
        fontsize=9, color=_ACCENT,
        arrowprops=dict(arrowstyle="->", color=_ACCENT, lw=1.2),
    )
    _style(
        ax,
        "1 · Implied volatility is a price, restated",
        "volatility assumption (%)",
        "Black-Scholes option price ($)",
    )
    # Bottom-right: the area under the curve is empty at high vol.
    ax.text(
        0.97, 0.04,
        "Price rises with volatility, always.\nSo one price ⇒ exactly one volatility.\n"
        "'Implied vol' is just the quote in\ndifferent units — like a bond yield.",
        transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=_GREY, alpha=0.9),
    )

    ax = axes[1]
    strikes = np.linspace(50, 170, 400)
    vega = np.asarray(bs_vega(S, strikes, T, r, 0.0, 0.25), dtype=float)
    ax.plot(strikes, vega, color=_GREEN, lw=2.2)
    ax.fill_between(strikes, 0, vega, color=_GREEN, alpha=0.12)
    ax.axvline(S, color=_GREY, lw=1.0, ls=":")
    ax.text(S + 1.5, vega.max() * 0.42, "at the money", fontsize=8, color=_GREY)

    for x, label in ((62, "deep in\nthe money"), (152, "far out of\nthe money")):
        ax.annotate(
            label, xy=(x, float(np.interp(x, strikes, vega))), xytext=(x, vega.max() * 0.30),
            fontsize=8, color=_ACCENT, ha="center",
            arrowprops=dict(arrowstyle="->", color=_ACCENT, lw=1.0),
        )

    _style(
        ax,
        "2 · …but the inversion is only well conditioned near the money",
        "strike ($), spot = 100",
        "vega — price sensitivity to volatility",
    )
    ax.text(
        0.03, 0.95,
        "Vega is the slope we invert. Where it\nvanishes, a one-cent quote error moves\n"
        "implied vol by whole points — which is\nwhy those quotes are filtered out, not fitted.",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=_GREY, alpha=0.9),
    )

    fig.suptitle(
        "What the solver is actually doing", fontsize=13, y=1.02, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# 2. Black-Scholes' flat assumption vs what the market says
# ---------------------------------------------------------------------------


def explain_flat_vs_surface(path: Path) -> Path:
    """The central point of the whole project, in three panels."""
    k = np.linspace(-0.45, 0.30, 300)
    iv = np.asarray(svi_implied_vol(k, _T, _REF), dtype=float) * 100
    atm = float(svi_implied_vol(0.0, _T, _REF)) * 100
    lo, hi = iv.min() - 6, iv.max() + 5

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # --- Panel A: what Black-Scholes assumes ---
    ax = axes[0]
    ax.plot(k, np.full_like(k, atm), color=_GREY, lw=2.4, ls="--")
    ax.set_ylim(lo, hi)
    _style(
        ax,
        "A · What Black-Scholes assumes",
        "strike, relative to the forward",
        "implied volatility (%)",
    )
    ax.text(
        0.5, 0.80,
        "One volatility for every strike.\nThe smile would be a flat line.",
        transform=ax.transAxes, fontsize=9, ha="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f2f3f5", ec=_GREY),
    )
    ax.set_xticks([-0.4, -0.2, 0.0, 0.2])
    ax.set_xticklabels(["−33%", "−18%", "at the\nmoney", "+22%"], fontsize=8)

    # --- Panel B: what the market actually quotes ---
    ax = axes[1]
    ax.plot(k, np.full_like(k, atm), color=_GREY, lw=1.6, ls="--", label="Black-Scholes")
    ax.plot(k, iv, color=_INK, lw=2.6, label="what the market quotes")
    ax.fill_between(k, np.full_like(k, atm), iv, where=iv > atm, color=_ACCENT, alpha=0.16)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    _style(
        ax,
        "B · What the market actually quotes",
        "strike, relative to the forward",
        "implied volatility (%)",
    )
    ax.annotate(
        "crash protection\ncosts much more",
        xy=(-0.30, float(np.interp(-0.30, k, iv))), xytext=(-0.13, hi - 4.5),
        fontsize=8.5, color=_ACCENT,
        arrowprops=dict(arrowstyle="->", color=_ACCENT, lw=1.2),
    )
    ax.set_xticks([-0.4, -0.2, 0.0, 0.2])
    ax.set_xticklabels(["−33%", "−18%", "at the\nmoney", "+22%"], fontsize=8)

    # --- Panel C: the dollar consequence ---
    ax = axes[2]
    F, T_ex, r_ex = 100.0, _T, 0.04
    k_hedge = float(np.log(0.90))
    iv_flat = float(svi_implied_vol(0.0, T_ex, _REF))
    iv_true = float(svi_implied_vol(k_hedge, T_ex, _REF))
    s_eff = F * np.exp(-r_ex * T_ex)
    p_flat = float(bs_put_price(s_eff, 90.0, T_ex, r_ex, 0.0, iv_flat))
    p_true = float(bs_put_price(s_eff, 90.0, T_ex, r_ex, 0.0, iv_true))

    bars = ax.bar(
        ["Black-Scholes\n(one flat vol)", "using the\nfitted surface"],
        [p_flat, p_true],
        color=[_GREY, _INK], width=0.55,
    )
    for rect, val, vol in zip(bars, (p_flat, p_true), (iv_flat, iv_true)):
        ax.text(
            rect.get_x() + rect.get_width() / 2, val + 0.06,
            f"${val:.2f}\n@ {vol * 100:.1f}% vol",
            ha="center", fontsize=9,
        )
    ax.set_ylim(0, max(p_flat, p_true) * 1.75)
    _style(ax, "C · Why it matters, in dollars", None, "price of one crash-hedge put ($)")
    ax.text(
        0.5, 0.78,
        f"Same option — a put 10% below the\nforward. Flat vol underprices the hedge\n"
        f"by {abs(p_true / p_flat - 1):.0%}.",
        transform=ax.transAxes, fontsize=8.5, ha="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=_ACCENT, alpha=0.95),
    )

    fig.suptitle(
        "The problem this project solves", fontsize=13, y=1.03, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# 3. What each SVI parameter does
# ---------------------------------------------------------------------------


def explain_svi_parameters(path: Path) -> Path:
    """Turn the five parameters into five pictures."""
    k = np.linspace(-0.6, 0.45, 400)

    variations = [
        ("a", "overall level", [0.004, 0.012, 0.022], "lifts or drops the whole smile"),
        ("b", "wing steepness", [0.06, 0.115, 0.20], "opens or closes the V"),
        ("rho", "skew / tilt", [-0.9, -0.62, 0.0], "rotates it — negative = crash fear"),
        ("m", "horizontal shift", [-0.10, 0.021, 0.14], "slides the low point sideways"),
        ("sigma", "curvature", [0.03, 0.135, 0.30], "rounds off the bottom"),
    ]

    symbols = {"rho": "ρ", "sigma": "σ", "a": "a", "b": "b", "m": "m"}

    # Precompute every curve so the shared y-axis can be sized from the data;
    # a hard-coded range clips the steeper wings.
    curves = {
        name: [
            np.asarray(
                svi_implied_vol(k, _T, SVIParams(**{**_REF.as_dict(), name: v})), dtype=float
            ) * 100
            for v in values
        ]
        for name, _, values, _ in variations
    }
    flat = np.concatenate([c for group in curves.values() for c in group])
    y_lo, y_hi = flat.min() - 3, flat.max() + 9  # headroom for the legends

    fig, axes = plt.subplots(1, 5, figsize=(19, 3.9), sharey=True)
    shades = ["#a9c0e0", _INK, "#0c1f3d"]

    for ax, (name, label, values, blurb) in zip(axes, variations):
        for value, colour, iv in zip(values, shades, curves[name]):
            ax.plot(k, iv, color=colour, lw=2.0, label=f"{symbols[name]} = {value:g}")

        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.set_ylim(y_lo, y_hi)
        symbol = symbols[name]
        _style(ax, f"{symbol}  —  {label}", "strike vs forward", None)
        ax.text(
            0.5, -0.30, blurb, transform=ax.transAxes, fontsize=8.5,
            ha="center", color="#444", style="italic",
        )

    axes[0].set_ylabel("implied volatility (%)", fontsize=9)
    fig.suptitle(
        "The five SVI parameters — each one bends the smile a different way",
        fontsize=13, y=1.06, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# 4. The data funnel
# ---------------------------------------------------------------------------


def explain_data_funnel(path: Path, stages: list[tuple[str, int]] | None = None) -> Path:
    """Show how much of a listed chain is unusable, and why."""
    if stages is None:
        stages = [
            ("Every listed SPY contract", 11169),
            ("Has a usable price", 5839),
            ("Inside the expiry window", 4857),
            ("Out-of-the-money, sane prices", 3000),
            ("Implied vol solved & plausible", 2939),
            ("Within 4 std devs of the forward", 2481),
            ("Not a local outlier", 2474),
            ("In one of the 12 fitted expiries", 1058),
        ]

    labels = [s[0] for s in stages]
    counts = np.array([s[1] for s in stages], dtype=float)
    top = counts[0]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    cmap = plt.get_cmap("viridis")

    for i, (label, count) in enumerate(zip(labels, counts)):
        width = count / top
        left = (1 - width) / 2
        ax.add_patch(
            Rectangle(
                (left, -i - 0.38), width, 0.76,
                facecolor=cmap(0.12 + 0.55 * i / max(len(stages) - 1, 1)),
                edgecolor="white", lw=1.4,
            )
        )
        ax.text(
            0.5, -i, f"{int(count):,}", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold",
        )
        ax.text(1.04, -i, label, ha="left", va="center", fontsize=9.5)
        if i:
            dropped = counts[i - 1] - count
            if dropped > 0:
                ax.text(
                    -0.04, -i + 0.5, f"−{int(dropped):,}",
                    ha="right", va="center", fontsize=8, color=_ACCENT,
                )

    ax.set_xlim(-0.30, 1.75)
    ax.set_ylim(-len(stages) + 0.3, 1.9)
    ax.axis("off")
    ax.text(
        0.5, 1.55,
        "Only ~9% of a listed options chain carries usable volatility information",
        ha="center", fontsize=12, fontweight="bold",
    )
    ax.text(
        0.5, 1.15,
        "zero bids · one-tick markets · contracts that have not traded in weeks · "
        "deep in-the-money quotes that are almost all intrinsic value",
        ha="center", fontsize=8.5, color="#555", style="italic",
    )
    ax.text(
        -0.04, -len(stages) + 0.75, "dropped", ha="right", fontsize=8,
        color=_ACCENT, style="italic",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# 5. Why in-the-money quotes are poison
# ---------------------------------------------------------------------------


def explain_why_otm(path: Path) -> Path:
    """Justify the single most consequential design decision in the project.

    A deep in-the-money option's price is almost entirely intrinsic value;
    only the sliver of time value carries volatility information. The same
    one-cent quote error therefore translates into a vastly larger implied
    vol error than it would out of the money.
    """
    F, T, r, vol = 100.0, 0.5, 0.04, 0.25
    strikes = np.linspace(60, 100, 200)
    s_eff = F * np.exp(-r * T)

    price = np.asarray(bs_call_price(s_eff, strikes, T, r, 0.0, vol), dtype=float)
    intrinsic = np.maximum(s_eff - strikes * np.exp(-r * T), 0.0)
    time_value = price - intrinsic

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    ax = axes[0]
    ax.fill_between(strikes, 0, intrinsic, color=_GREY, alpha=0.45, label="intrinsic value — no vol information")
    ax.fill_between(strikes, intrinsic, price, color=_GREEN, alpha=0.65, label="time value — the only part that reveals vol")
    ax.plot(strikes, price, color=_INK, lw=1.8, label="quoted call price")
    ax.axvline(100, color=_GREY, ls=":", lw=1.0)
    ax.text(99.2, price.max() * 0.45, "at the money", rotation=90, fontsize=8,
            color=_GREY, ha="right", va="center")
    _style(ax, "A deep in-the-money call is almost all intrinsic", "strike ($), forward = 100", "price ($)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax = axes[1]
    # Implied-vol error produced by a one-cent price error, = 0.01 / vega.
    vega = np.asarray(bs_vega(s_eff, strikes, T, r, 0.0, vol), dtype=float)
    iv_err = 0.01 / np.maximum(vega, 1e-9) * 100
    ax.plot(strikes, iv_err, color=_ACCENT, lw=2.4)
    ax.set_yscale("log")
    ax.axvline(100, color=_GREY, ls=":", lw=1.0)
    _style(
        ax,
        "…so a 1-cent quote error explodes into vol error",
        "strike ($), forward = 100",
        "implied vol error from 1¢ (vol points, log scale)",
    )
    ax.text(
        0.04, 0.93,
        "This is why the pipeline uses only\nout-of-the-money options.\n\n"
        "An early version kept in-the-money\nquotes as a fallback and produced\n"
        "fit errors of 1,500+ basis points.",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=_ACCENT, alpha=0.95),
    )

    fig.suptitle(
        "Why only out-of-the-money quotes are used",
        fontsize=13, y=1.02, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# 6. Total variance and the calendar-arbitrage condition
# ---------------------------------------------------------------------------


def explain_arbitrage_checks(path: Path) -> Path:
    """Explain the two arbitrage conditions in pictures rather than algebra."""
    k = np.linspace(-0.5, 0.4, 400)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    # --- Butterfly ---
    ax = axes[0]
    good = _REF
    bad = SVIParams(a=0.02, b=0.70, rho=-0.85, m=0.0, sigma=0.18)
    for params, colour, label, ls in (
        (good, _GREEN, "a healthy slice", "-"),
        (bad, _ACCENT, "an arbitrageable slice", "--"),
    ):
        from .svi import durrleman_g

        g = np.asarray(durrleman_g(k, params), dtype=float)
        ax.plot(k, g, color=colour, lw=2.2, ls=ls, label=label)

    ax.axhline(0, color="black", lw=1.2)
    ax.fill_between(k, -2, 0, color=_ACCENT, alpha=0.08)
    ax.set_ylim(-0.85, 2.75)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    _style(ax, "Butterfly check — the density must stay positive", "strike vs forward", "Durrleman g(k)")
    ax.text(
        0.03, 0.06,
        "Below zero means the fitted smile implies a\nnegative probability for some outcome —\n"
        "free money for anyone who spots it.",
        transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=_GREY, alpha=0.92),
    )

    # --- Calendar ---
    ax = axes[1]
    near = SVIParams(a=0.008, b=0.075, rho=-0.62, m=0.02, sigma=0.09)
    far = SVIParams(a=0.026, b=0.150, rho=-0.60, m=0.03, sigma=0.18)
    crossing = SVIParams(a=0.004, b=0.060, rho=-0.60, m=0.03, sigma=0.07)

    ax.plot(k, svi_total_variance(k, near), color=_INK, lw=2.2, label="3 months")
    ax.plot(k, svi_total_variance(k, far), color=_GREEN, lw=2.2, label="1 year — correctly above")
    ax.plot(k, svi_total_variance(k, crossing), color=_ACCENT, lw=2.0, ls="--",
            label="1 year — impossible, dips below")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.set_ylim(-0.055, 0.175)
    _style(ax, "Calendar check — total variance must grow with time", "strike vs forward", "total variance  w = σ²T")
    ax.text(
        0.03, 0.06,
        "A longer-dated option cannot imply less\ntotal uncertainty than a shorter one.\n"
        "If it did, the calendar spread is riskless profit.",
        transform=ax.transAxes, fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=_GREY, alpha=0.92),
    )

    fig.suptitle(
        "The two sanity checks every fitted slice must pass",
        fontsize=13, y=1.02, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------


def make_all_explainers(figures_dir: Path) -> list[Path]:
    """Generate the whole explainer set into ``figures_dir``."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [
        explain_flat_vs_surface(figures_dir / "explainer_problem.png"),
        explain_implied_vol(figures_dir / "explainer_implied_vol.png"),
        explain_why_otm(figures_dir / "explainer_why_otm.png"),
        explain_svi_parameters(figures_dir / "explainer_svi_params.png"),
        explain_data_funnel(figures_dir / "explainer_data_funnel.png"),
        explain_arbitrage_checks(figures_dir / "explainer_arbitrage.png"),
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="volsurface.explainers",
        description="Generate the README explainer figures.",
    )
    p.add_argument("--figures-dir", type=Path, default=Path.cwd() / "figures")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = make_all_explainers(args.figures_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
