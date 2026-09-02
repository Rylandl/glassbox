"""End-to-end smoke coverage for every leaf of the ``glassbox`` command tree.

The tree is read from the dispatcher itself, so a command added to
:mod:`glassbox.cli._tree` is swept here automatically. Each leaf is invoked
with ``--help`` only, so argparse prints usage and exits with status zero
before any fit, simulation, or file I/O happens.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from glassbox import cli
from glassbox.cli._tree import Command, Group

_PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"

_LEAF_PATHS = cli.leaf_paths()
_LEAF_IDS = [" ".join(path) for path in _LEAF_PATHS]

# Third-party packages and the Glassbox modules that reach for them. Blocking
# these proves that rendering the command tree imports no workflow module.
_OPTIONAL_MODULES = (
    "crazyflow",
    "cascade",
    "glassbox.integrations.cascade",
    "glassbox.integrations.crazyflow",
    "glassbox.integrations.crazyflow_animation",
    "glassbox.integrations.crazyflow_bootstrap",
    "glassbox.integrations.crazyflow_prototype",
    "glassbox.integrations.crazyflow_throw",
)


@pytest.mark.parametrize("path", _LEAF_PATHS, ids=_LEAF_IDS)
def test_subcommand_help_exits_cleanly(
    path: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([*path, "--help"])

    code = excinfo.value.code
    if isinstance(code, str) and "needs the optional" in code:
        pytest.skip(code.splitlines()[0])
    assert code == 0
    stdout = capsys.readouterr().out
    assert "usage:" in stdout
    assert stdout.startswith(f"usage: glassbox {' '.join(path)}")


def test_every_leaf_path_resolves_to_a_command() -> None:
    assert _LEAF_PATHS
    for path in _LEAF_PATHS:
        node = cli.find(path)
        assert isinstance(node, Command)
        module_name, separator, function_name = node.target.partition(":")
        assert module_name and separator == ":" and function_name


def test_top_level_help_lists_every_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])

    assert excinfo.value.code == 0
    stdout = capsys.readouterr().out
    for node in cli.TREE:
        assert node.name in stdout
        children = (
            [item.name for item in node.commands]
            if isinstance(node, Group)
            else list(node.subcommands)
        )
        for child in children:
            assert child in stdout


def test_top_level_help_runs_without_any_optional_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in _OPTIONAL_MODULES:
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])

    assert excinfo.value.code == 0
    assert "crazyflow" in capsys.readouterr().out


def test_a_command_needing_a_missing_extra_reports_it_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "glassbox.integrations.crazyflow_throw", None)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["crazyflow", "throw", "--help"])

    message = excinfo.value.code
    assert isinstance(message, str)
    assert "glassbox crazyflow throw needs the optional 'crazyflow' extra" in message
    assert "uv run --extra crazyflow" in message


@pytest.mark.parametrize(
    "argv",
    ([], ["not-a-command"], ["crazyflow"], ["crazyflow", "not-a-command"]),
    ids=["no-command", "unknown-command", "group-without-command", "unknown-in-group"],
)
def test_missing_or_unknown_commands_exit_two(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)

    assert excinfo.value.code == 2


def test_pyproject_declares_exactly_one_console_script() -> None:
    with _PYPROJECT_PATH.open("rb") as handle:
        data = tomllib.load(handle)

    assert data["project"]["scripts"] == {"glassbox": "glassbox.cli:main"}
