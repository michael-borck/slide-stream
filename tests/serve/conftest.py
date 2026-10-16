"""Shared fixtures for the web UI tests: a fresh config, and module state
(jobs, demo rate-limit hits, projects) cleared around every test."""

import copy

import pytest

pytest.importorskip("fastapi")

from slide_stream import serve  # noqa: E402
from slide_stream.config_loader import DEFAULT_CONFIG  # noqa: E402


@pytest.fixture
def base_config():
    return copy.deepcopy(DEFAULT_CONFIG)


@pytest.fixture(autouse=True)
def _clear_jobs():
    serve._JOBS.clear()
    yield
    serve._JOBS.clear()


@pytest.fixture(autouse=True)
def _clear_demo_hits():
    serve._DEMO_HITS.clear()
    serve._DEMO_DRAFT_HITS.clear()
    yield
    serve._DEMO_HITS.clear()
    serve._DEMO_DRAFT_HITS.clear()


@pytest.fixture(autouse=True)
def _clear_projects():
    serve._PROJECTS.clear()
    yield
    serve._PROJECTS.clear()
