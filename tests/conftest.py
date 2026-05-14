"""Test fixtures for Stride unit and integration tests.

Key fixtures:
- config_no_dirs: StriderConfig that doesn't create directories (for unit tests)
- tmp_output_dir: Temporary directory for test output
"""

import tempfile
from pathlib import Path

import pytest

from stride.config import StriderConfig


@pytest.fixture
def config_no_dirs():
    """Return a StriderConfig that doesn't create directories.

    This fixture is essential for unit tests — it prevents the config from
    polluting the filesystem during test instantiation. Call
    config.ensure_directories() explicitly only in integration tests that
    actually need it.
    """
    return StriderConfig(
        model_dir=Path("models"),  # Not created
        output_dir=Path("output"),  # Not created
    )


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Return a temporary directory for test output files.

    Each test gets its own isolated tmp directory, automatically cleaned up
    after the test completes.
    """
    return tmp_path / "stride_test_output"


@pytest.fixture
def config_with_tmp_output(config_no_dirs, tmp_output_dir):
    """Return a StriderConfig with output_dir set to a temporary directory.

    Useful for integration tests that need to write output without polluting
    the real filesystem.
    """
    config = config_no_dirs
    config.output_dir = tmp_output_dir
    return config
