"""Options-chain retrieval and on-disk caching.

Market data is fetched once per ticker per calendar day and written to
parquet, so re-running the pipeline (or running the tests, or regenerating a
figure) is reproducible and does not re-hit the network.  A ``meta.json``
sidecar records the spot price and the exact timestamp of the pull, which is
what makes a cached run genuinely reproducible rather than merely fast.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

__all__ = ["ChainSnapshot", "fetch_chain", "load_cached", "list_cached", "CACHE_COLUMNS"]

#: Columns kept from the raw yfinance chain.  Anything else is presentation.
CACHE_COLUMNS = [
    "contractSymbol",
    "expiry",
    "option_type",
    "strike",
    "bid",
    "ask",
    "lastPrice",
    "volume",
    "openInterest",
    "lastTradeDate",
    "inTheMoney",
]


@dataclass
class ChainSnapshot:
    """A point-in-time options chain plus the underlying state it goes with."""

    ticker: str
    spot: float
    asof: datetime
    """UTC timestamp of the data pull."""

    chain: pd.DataFrame
    """Long-format chain: one row per contract, columns per :data:`CACHE_COLUMNS`."""

    source: str
    """``"network"`` or ``"cache"`` -- surfaced in the run log for provenance."""

    @property
    def n_contracts(self) -> int:
        return len(self.chain)

    @property
    def expiries(self) -> list[str]:
        return sorted(self.chain["expiry"].unique().tolist())


def _cache_paths(cfg: Config, ticker: str, day: str) -> tuple[Path, Path]:
    base = cfg.cache_dir / ticker.upper() / day
    return base / "chain.parquet", base / "meta.json"


def list_cached(cfg: Config, ticker: str) -> list[str]:
    """Return cached snapshot days (newest first) for ``ticker``."""
    base = cfg.cache_dir / ticker.upper()
    if not base.exists():
        return []
    days = [p.name for p in base.iterdir() if (p / "chain.parquet").exists()]
    return sorted(days, reverse=True)


def load_cached(cfg: Config, ticker: str, day: str | None = None) -> ChainSnapshot | None:
    """Load a cached snapshot.  ``day=None`` loads the most recent one."""
    if day is None:
        days = list_cached(cfg, ticker)
        if not days:
            return None
        day = days[0]

    chain_path, meta_path = _cache_paths(cfg, ticker, day)
    if not (chain_path.exists() and meta_path.exists()):
        return None

    chain = pd.read_parquet(chain_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return ChainSnapshot(
        ticker=ticker.upper(),
        spot=float(meta["spot"]),
        asof=datetime.fromisoformat(meta["asof"]),
        chain=chain,
        source="cache",
    )


def _save_cache(cfg: Config, snap: ChainSnapshot) -> None:
    day = snap.asof.strftime("%Y-%m-%d")
    chain_path, meta_path = _cache_paths(cfg, snap.ticker, day)
    chain_path.parent.mkdir(parents=True, exist_ok=True)
    snap.chain.to_parquet(chain_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "ticker": snap.ticker,
                "spot": snap.spot,
                "asof": snap.asof.isoformat(),
                "n_contracts": int(len(snap.chain)),
                "expiries": snap.expiries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("cached %d contracts -> %s", len(snap.chain), chain_path)


def _get_spot(tk) -> float:
    """Spot price, preferring the live quote and falling back to daily bars."""
    try:
        price = tk.fast_info.get("lastPrice")
        if price and float(price) > 0:
            return float(price)
    except Exception:  # pragma: no cover - network/library variability
        log.debug("fast_info unavailable, falling back to history", exc_info=True)

    hist = tk.history(period="5d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError("could not determine spot price")
    return float(hist["Close"].iloc[-1])


def fetch_chain(cfg: Config, ticker: str | None = None) -> ChainSnapshot:
    """Get an options chain, from cache when possible and the network otherwise.

    Raises
    ------
    RuntimeError
        If the network is unavailable (or disabled via ``cfg.offline``) and
        no cached snapshot exists.
    """
    ticker = (ticker or cfg.ticker).upper()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if cfg.use_cache and not cfg.refresh:
        cached = load_cached(cfg, ticker, today)
        if cached is not None:
            log.info("using cached snapshot for %s on %s", ticker, today)
            return cached

    if cfg.offline:
        cached = load_cached(cfg, ticker)
        if cached is None:
            raise RuntimeError(
                f"offline mode requested but no cached chain exists for {ticker}"
            )
        log.info("offline: using cached snapshot from %s", cached.asof.date())
        return cached

    try:
        snap = _fetch_from_network(ticker)
    except Exception as exc:
        cached = load_cached(cfg, ticker)
        if cached is None:
            raise RuntimeError(
                f"failed to fetch options chain for {ticker} and no cache is available: {exc}"
            ) from exc
        log.warning("network fetch failed (%s); falling back to cache from %s", exc, cached.asof.date())
        return cached

    if cfg.use_cache:
        _save_cache(cfg, snap)
    return snap


def _fetch_from_network(ticker: str) -> ChainSnapshot:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = _get_spot(tk)

    expiries = list(tk.options)
    if not expiries:
        raise RuntimeError(f"no listed option expiries returned for {ticker}")

    log.info("fetching %d expiries for %s (spot %.2f)", len(expiries), ticker, spot)

    frames: list[pd.DataFrame] = []
    for expiry in expiries:
        try:
            chain = tk.option_chain(expiry)
        except Exception:  # pragma: no cover - a single flaky expiry
            log.warning("skipping expiry %s: chain fetch failed", expiry, exc_info=True)
            continue

        for side, df in (("call", chain.calls), ("put", chain.puts)):
            if df is None or df.empty:
                continue
            part = df.copy()
            part["expiry"] = expiry
            part["option_type"] = side
            missing = [c for c in CACHE_COLUMNS if c not in part.columns]
            for col in missing:
                part[col] = pd.NA
            frames.append(part[CACHE_COLUMNS])

    if not frames:
        raise RuntimeError(f"every expiry fetch failed for {ticker}")

    full = pd.concat(frames, ignore_index=True)
    # Normalise dtypes so the parquet round-trip is lossless and downstream
    # numeric code never sees object columns.
    for col in ("strike", "bid", "ask", "lastPrice", "volume", "openInterest"):
        full[col] = pd.to_numeric(full[col], errors="coerce")
    full["lastTradeDate"] = pd.to_datetime(full["lastTradeDate"], errors="coerce", utc=True)

    return ChainSnapshot(
        ticker=ticker,
        spot=spot,
        asof=datetime.now(timezone.utc),
        chain=full,
        source="network",
    )
