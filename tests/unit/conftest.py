"""Pytest configuration and fixtures for unit tests."""

import pytest

from stride.config import StriderConfig


@pytest.fixture
def config():
    """Provide a default config for tests."""
    return StriderConfig()
