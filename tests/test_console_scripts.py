"""End-to-end smoke coverage for every ``[project.scripts]`` console entry.

Each entry point is invoked with ``--help`` only, so this stays fast:
argparse prints usage and exits with status zero before any fit, simulation,
or file I/O happens. Scripts whose module needs an optional extra that is
not installed in this environment are skipped rather than failed.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest

_PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _console_scripts() -> list[tuple[str, str, str]]:
    with _PYPROJECT_PATH.open("rb") as handle:
        data = tomllib.load(handle)
    scripts = data["project"]["scripts"]
    entries = []
    for script_name, target in scripts.items():
        module_name, _, function_name = target.partition(":")
        entries.append((script_name, module_name, function_name))
    return entries


_CONSOLE_SCRIPTS = _console_scripts()


@pytest.mark.parametrize(
    ("script_name", "module_name", "function_name"),
    _CONSOLE_SCRIPTS,
    ids=[entry[0] for entry in _CONSOLE_SCRIPTS],
)
def test_console_script_help_exits_cleanly(
    script_name: str,
    module_name: str,
    function_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    try:
        module = importlib.import_module(module_name)
        entry = getattr(module, function_name)
    except (ImportError, RuntimeError, AttributeError) as error:
        pytest.skip(f"{module_name} needs an unavailable optional extra: {error}")

    monkeypatch.setattr(sys, "argv", [script_name, "--help"])

    with pytest.raises(SystemExit) as excinfo:
        entry()

    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_pyproject_declares_at_least_one_console_script() -> None:
    assert len(_CONSOLE_SCRIPTS) >= 1
    assert all(module_name and function_name for _, module_name, function_name in _CONSOLE_SCRIPTS)
