"""Command-line entry point: ``python -m volsurface --ticker SPY --rate 0.04``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volsurface",
        description=(
            "Build an implied-volatility surface from a live options chain and "
            "calibrate a raw SVI slice to each expiry."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="example:  python -m volsurface --ticker SPY --rate 0.04",
    )

    g = p.add_argument_group("market")
    g.add_argument("--ticker", default="SPY", help="underlying symbol")
    g.add_argument("--rate", type=float, default=0.04, help="continuous risk-free rate")
    g.add_argument(
        "--dividend-yield",
        type=float,
        default=0.012,
        help="continuous dividend yield (fallback only when the forward cannot be implied)",
    )
    g.add_argument(
        "--no-imply-forward",
        action="store_true",
        help="use spot*exp((r-q)T) instead of recovering the forward from put-call parity",
    )

    g = p.add_argument_group("expiry selection")
    g.add_argument("--min-days", type=int, default=7, help="drop expiries nearer than this")
    g.add_argument("--max-days", type=int, default=550, help="drop expiries beyond this")
    g.add_argument("--max-expiries", type=int, default=12, help="cap on slices to fit")

    g = p.add_argument_group("quote filters")
    g.add_argument(
        "--min-price", type=float, default=0.05, help="minimum acceptable option premium"
    )
    g.add_argument(
        "--max-rel-spread", type=float, default=0.60, help="max (ask-bid)/mid"
    )
    g.add_argument(
        "--no-last-trade",
        action="store_true",
        help="require a live two-sided quote; do not fall back to the last traded price",
    )
    g.add_argument(
        "--max-trade-age",
        type=int,
        default=1,
        metavar="SESSIONS",
        help="how many sessions back a last trade may be and still count as fresh",
    )
    g.add_argument(
        "--max-moneyness",
        type=float,
        default=1.2,
        help="hard cap on |log(K/F)|",
    )
    g.add_argument(
        "--max-sd",
        type=float,
        default=4.0,
        help="adaptive wing cutoff in standard deviations (iv*sqrt(T)) from the forward",
    )
    g.add_argument(
        "--min-points", type=int, default=8, help="minimum clean quotes to fit a slice"
    )
    g.add_argument(
        "--min-wing-points",
        type=int,
        default=3,
        help="minimum points required on each side of the forward",
    )

    g = p.add_argument_group("fitting")
    g.add_argument(
        "--no-vega-weight",
        action="store_true",
        help="fit with equal weights instead of vega weighting",
    )
    g.add_argument("--n-starts", type=int, default=24, help="multi-start count per slice")
    g.add_argument("--seed", type=int, default=7, help="seed for the random restarts")

    g = p.add_argument_group("io")
    g.add_argument("--root", type=Path, default=None, help="project root for outputs")
    g.add_argument("--refresh", action="store_true", help="force a fresh network pull")
    g.add_argument(
        "--offline", action="store_true", help="use the cache only; never hit the network"
    )
    g.add_argument("--no-cache", action="store_true", help="do not read or write the cache")
    g.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    g.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")

    return p


def _config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        ticker=args.ticker.upper(),
        rate=args.rate,
        dividend_yield=args.dividend_yield,
        imply_forward=not args.no_imply_forward,
        min_days=args.min_days,
        max_days=args.max_days,
        max_expiries=args.max_expiries,
        min_price=args.min_price,
        max_rel_spread=args.max_rel_spread,
        allow_last_trade=not args.no_last_trade,
        max_trade_age_sessions=args.max_trade_age,
        max_abs_log_moneyness=args.max_moneyness,
        max_sd_moneyness=args.max_sd,
        min_points_per_slice=args.min_points,
        min_points_per_wing=args.min_wing_points,
        vega_weighted=not args.no_vega_weight,
        n_starts=args.n_starts,
        seed=args.seed,
        root=args.root or Path.cwd(),
        use_cache=not args.no_cache,
        refresh=args.refresh,
        offline=args.offline,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # yfinance is chatty about progress bars and retries.
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    logging.getLogger("peewee").setLevel(logging.ERROR)

    cfg = _config_from_args(args)

    # Imported here so that --help stays fast and does not pull in matplotlib.
    from .pipeline import run_pipeline

    try:
        result = run_pipeline(cfg)
    except (RuntimeError, ValueError) as exc:
        print(f"volsurface: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("volsurface: interrupted", file=sys.stderr)
        return 130

    print()
    print("=" * 68)
    print(result.summary())
    print("=" * 68)
    print("\nfigures:")
    for path in result.figures:
        print(f"  {path}")
    print("\noutputs:")
    for name, path in result.outputs.items():
        print(f"  {name:16s} {path}")
    print()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
