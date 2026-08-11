"""A synthetic options chain generated from a known SVI surface.

Building the test chain from a surface we choose gives the whole pipeline a
ground truth to be checked against.  If the code mis-handles the forward,
picks the wrong option side, or mangles a day count, the recovered implied
vols stop matching the ones we priced with -- a far sharper test than
asserting that the output "looks reasonable".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from volsurface.black_scholes import bs_call_price, bs_put_price
from volsurface.data import CACHE_COLUMNS, ChainSnapshot
from volsurface.surface import year_fraction
from volsurface.svi import SVIParams, svi_total_variance

__all__ = [
    "ASOF",
    "SPOT",
    "RATE",
    "DIV_YIELD",
    "SLICES",
    "SyntheticChain",
    "build_chain",
]

ASOF = datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc)
SPOT = 500.0
RATE = 0.04
DIV_YIELD = 0.01

#: Expiries in calendar days from :data:`ASOF`, and the SVI slice for each.
#: Total variance increases with maturity, so the surface is
#: calendar-arbitrage free by construction.
SLICES: list[tuple[int, SVIParams]] = [
    (30, SVIParams(a=0.0016, b=0.030, rho=-0.70, m=0.010, sigma=0.050)),
    (60, SVIParams(a=0.0032, b=0.048, rho=-0.68, m=0.014, sigma=0.070)),
    (120, SVIParams(a=0.0065, b=0.075, rho=-0.65, m=0.020, sigma=0.100)),
    (240, SVIParams(a=0.0130, b=0.115, rho=-0.60, m=0.028, sigma=0.150)),
    (360, SVIParams(a=0.0200, b=0.150, rho=-0.55, m=0.035, sigma=0.190)),
]


@dataclass
class SyntheticChain:
    """A generated chain plus the truth used to build it."""

    snapshot: ChainSnapshot
    truth: dict[str, SVIParams]
    """Expiry -> the SVI parameters its prices were generated from."""

    forwards: dict[str, float]
    maturities: dict[str, float]

    def true_iv(self, expiry: str, k: float) -> float:
        T = self.maturities[expiry]
        return float(np.sqrt(svi_total_variance(k, self.truth[expiry]) / T))


def build_chain(
    *,
    spread_frac: float = 0.02,
    two_sided: bool = True,
    stale_days: int | None = None,
    n_strikes: int = 41,
    sd_span: float = 2.5,
    noise_bps: float = 0.0,
    seed: int = 0,
) -> SyntheticChain:
    """Generate a chain.  ``noise_bps`` adds vol noise before pricing."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    truth: dict[str, SVIParams] = {}
    forwards: dict[str, float] = {}
    maturities: dict[str, float] = {}

    for days, params in SLICES:
        expiry = (ASOF + timedelta(days=days)).strftime("%Y-%m-%d")
        T = year_fraction(expiry, ASOF)
        forward = SPOT * np.exp((RATE - DIV_YIELD) * T)
        s_eff = forward * np.exp(-RATE * T)

        truth[expiry] = params
        forwards[expiry] = forward
        maturities[expiry] = T

        atm_vol = float(np.sqrt(svi_total_variance(0.0, params) / T))
        k_max = sd_span * atm_vol * np.sqrt(T)
        ks = np.linspace(-k_max, k_max, n_strikes)

        traded = ASOF if stale_days is None else ASOF - timedelta(days=stale_days)

        for k in ks:
            strike = float(forward * np.exp(k))
            iv = float(np.sqrt(svi_total_variance(k, params) / T))
            if noise_bps:
                iv = max(iv + rng.normal(0.0, noise_bps * 1e-4), 1e-4)

            for side, fn in (("call", bs_call_price), ("put", bs_put_price)):
                price = float(fn(s_eff, strike, T, RATE, 0.0, iv))
                half = 0.5 * spread_frac * price
                rows.append(
                    {
                        "contractSymbol": f"SYN{expiry}{side[0].upper()}{strike:012.4f}",
                        "expiry": expiry,
                        "option_type": side,
                        "strike": strike,
                        "bid": max(price - half, 0.0) if two_sided else 0.0,
                        "ask": price + half if two_sided else 0.0,
                        "lastPrice": price,
                        "volume": 100.0,
                        "openInterest": 500,
                        "lastTradeDate": traded,
                        "inTheMoney": (side == "call" and strike < forward)
                        or (side == "put" and strike > forward),
                    }
                )

    chain = pd.DataFrame(rows)[CACHE_COLUMNS]
    chain["lastTradeDate"] = pd.to_datetime(chain["lastTradeDate"], utc=True)

    snapshot = ChainSnapshot(
        ticker="SYN", spot=SPOT, asof=ASOF, chain=chain, source="synthetic"
    )
    return SyntheticChain(snapshot, truth, forwards, maturities)
