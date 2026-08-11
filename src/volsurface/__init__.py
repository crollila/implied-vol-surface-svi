"""volsurface -- implied volatility surface construction and SVI calibration.

A self-contained pipeline that pulls a live equity-index options chain, backs
out Black-Scholes implied volatilities, assembles a clean surface, and fits
the raw SVI parameterisation to each expiry slice with arbitrage diagnostics.

Typical use::

    from volsurface import Config, run_pipeline

    result = run_pipeline(Config(ticker="SPY", rate=0.04))
    print(result.params_table)

or from the command line::

    python -m volsurface --ticker SPY --rate 0.04
"""

from __future__ import annotations

from .black_scholes import bs_call_price, bs_price, bs_put_price, bs_vega
from .config import Config
from .implied_vol import implied_vol, implied_vol_vectorised
from .surface import SurfaceData, build_surface
from .svi import (
    SVIFit,
    SVIParams,
    check_butterfly,
    check_calendar_arbitrage,
    durrleman_g,
    fit_slice,
    svi_implied_vol,
    svi_total_variance,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Config",
    "bs_price",
    "bs_call_price",
    "bs_put_price",
    "bs_vega",
    "implied_vol",
    "implied_vol_vectorised",
    "build_surface",
    "SurfaceData",
    "SVIParams",
    "SVIFit",
    "fit_slice",
    "svi_total_variance",
    "svi_implied_vol",
    "durrleman_g",
    "check_butterfly",
    "check_calendar_arbitrage",
    "run_pipeline",
    "PipelineResult",
]


def __getattr__(name: str):
    """Lazily expose the pipeline so importing the package stays cheap.

    ``run_pipeline`` pulls in matplotlib and yfinance; the numerical core
    should be importable without either.
    """
    if name in {"run_pipeline", "PipelineResult"}:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
