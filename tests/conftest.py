"""Shared pytest fixtures."""

import pytest

from study_assistant.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Make sure env overrides in one test never leak into another via the lru_cache."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
