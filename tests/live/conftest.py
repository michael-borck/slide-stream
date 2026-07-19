"""Fixtures for the live e2e suite (real, configured providers)."""

import copy

import pytest

from slide_stream.config_loader import load_config


@pytest.fixture
def live_config():
    """The real layered config (~/.slidestream.yaml + ./slidestream.yaml),
    a fresh deep copy per test so a test can mutate it freely."""
    return copy.deepcopy(load_config())
