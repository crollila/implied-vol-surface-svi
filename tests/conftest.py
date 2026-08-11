"""Pytest fixtures wrapping the synthetic chain generator in ``synthetic.py``."""

from __future__ import annotations

import pytest

from synthetic import SyntheticChain, build_chain


@pytest.fixture
def synthetic_chain() -> SyntheticChain:
    """Fully quoted chain: tight two-sided markets, all fresh."""
    return build_chain()


@pytest.fixture
def quoteless_chain() -> SyntheticChain:
    """No live quotes at all -- only same-session last trades.

    This is what Yahoo actually returns outside market hours, and the case
    the last-trade fallback exists for.
    """
    return build_chain(two_sided=False)


@pytest.fixture
def stale_chain() -> SyntheticChain:
    """No live quotes, and the last trades are two weeks old."""
    return build_chain(two_sided=False, stale_days=14)


@pytest.fixture
def chain_builder():
    """The generator itself, for tests needing custom chain parameters."""
    return build_chain
