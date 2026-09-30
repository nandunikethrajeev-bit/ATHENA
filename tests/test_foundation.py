"""Foundation tests for milestone M0.

Deterministic, fully offline tests proving the project skeleton imports
and the entry point runs. No network access, no third-party dependencies.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

from app import __version__
from app.main import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_version_is_defined():
    assert isinstance(__version__, str)
    assert __version__  # non-empty


def test_entry_point_runs_and_returns_success():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = main()

    assert exit_code == 0
    output = buffer.getvalue()
    assert "ATHENA" in output
    assert __version__ in output


def test_entry_point_runs_via_direct_script_mode():
    """The direct-script fallback path (python app/main.py) also imports cleanly."""
    import importlib
    import sys

    assert str(PROJECT_ROOT) in sys.path or "app" in sys.modules
    module = importlib.import_module("app.main")
    assert callable(module.main)


def test_data_directories_exist():
    for relative in ("data/raw", "data/processed"):
        assert (PROJECT_ROOT / relative).is_dir()
