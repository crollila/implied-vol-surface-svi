"""Run configuration for the vol-surface pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Config", "DEFAULT_ROOT"]

DEFAULT_ROOT = Path.cwd()


@dataclass
class Config:
    """Everything the pipeline needs to run, in one place.

    Defaults are tuned for SPY, whose chain is dense enough that fairly
    aggressive quote filtering still leaves plenty of points per slice.
    """

    ticker: str = "SPY"

    # --- Market inputs ----------------------------------------------------
    rate: float = 0.04
    """Continuously compounded risk-free rate."""

    dividend_yield: float = 0.012
    """Continuous dividend yield.  Only used as a fallback: by default the
    forward is implied from put-call parity, which absorbs the true carry."""

    imply_forward: bool = True
    """Estimate the per-expiry forward from put-call parity instead of
    trusting ``spot * exp((r - q) * T)``.  This is what makes the ATM level
    of the surface trustworthy without a dividend forecast."""

    # --- Expiry selection -------------------------------------------------
    min_days: int = 7
    """Drop expiries inside this many calendar days -- their IVs are
    dominated by microstructure and pin risk."""

    max_days: int = 550
    """Drop expiries beyond this horizon; LEAPS quotes are wide and stale."""

    max_expiries: int = 12
    """Cap on the number of expiry slices to fit, spread across the term
    structure."""

    # --- Quote filters ----------------------------------------------------
    min_price: float = 0.05
    """Minimum acceptable option premium.  Below a few ticks the quoted price
    is all rounding and the implied vol is noise."""

    max_rel_spread: float = 0.60
    """Maximum (ask - bid) / mid for a two-sided quote.  Wider than this and
    the mid is fiction."""

    allow_last_trade: bool = True
    """Fall back to the last traded price when no usable two-sided quote
    exists.  Essential outside market hours, when venues withdraw quotes and
    a chain would otherwise be empty."""

    max_trade_age_sessions: int = 1
    """How many trading sessions back a last trade may be and still count as
    fresh.  1 means "traded in the most recent session present in the data",
    which resolves correctly both intraday and after the close."""

    max_abs_log_moneyness: float = 1.2
    """Hard cap on |log(K/F)|, applied before implied vols are known."""

    max_sd_moneyness: float = 4.0
    """Adaptive wing cutoff: keep strikes within this many standard
    deviations (``iv * sqrt(T)``) of the forward.  Maturity-aware, unlike a
    fixed log-moneyness band."""

    min_points_per_slice: int = 8
    """Slices thinner than this cannot support a 5-parameter fit."""

    min_points_per_wing: int = 3
    """Points required strictly either side of the forward.  A one-winged
    slice cannot identify ``rho`` and ``m``, so it is dropped rather than
    fitted to a shape the data does not contain."""

    # --- Fitting ----------------------------------------------------------
    vega_weighted: bool = True
    """Weight the least-squares fit by BS vega, so the fit tracks where the
    market actually has price information rather than chasing wing noise."""

    n_starts: int = 24
    """Multi-start count for the SVI calibration."""

    # --- IO ---------------------------------------------------------------
    root: Path = field(default_factory=lambda: DEFAULT_ROOT)
    use_cache: bool = True
    refresh: bool = False
    """Force a network pull even if a cache entry exists for today."""

    offline: bool = False
    """Never hit the network; fail if no cache entry exists."""

    seed: int = 7

    @property
    def cache_dir(self) -> Path:
        return self.root / "data" / "cache"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    def ensure_dirs(self) -> None:
        for d in (self.cache_dir, self.figures_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)
