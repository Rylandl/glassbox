"""The single ``glassbox`` console command and its subcommand tree.

``glassbox <command> [...]`` dispatches to one workflow's own argparse front
end. The tree itself is static data in :mod:`glassbox.cli._tree`, so
``glassbox --help`` renders every command without importing a single workflow
module; a command that needs an optional extra only fails when it is run, and
then with the actionable message its own module raises.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from glassbox.cli._tree import TREE, Command, Group, Node, find, leaf_paths

__all__ = ["TREE", "Command", "Group", "find", "leaf_paths", "main"]

_DESCRIPTION = "Telemetry-driven differentiable vehicle dynamics identification."


def _program(path: Sequence[str]) -> str:
    return " ".join(("glassbox", *path))


def _nested_commands() -> str:
    lines = ["nested commands:"]
    for node in TREE:
        children = (
            [item.name for item in node.commands]
            if isinstance(node, Group)
            else list(node.subcommands)
        )
        if not children:
            continue
        lines.append(f"  {_program((node.name,))} {' | '.join(children)}")
    lines.append("")
    lines.append("Run 'glassbox <command> --help' for one command's own options.")
    return "\n".join(lines)


def _level_parser(
    path: Sequence[str], nodes: Sequence[Node], summary: str
) -> argparse.ArgumentParser:
    """Build the parser that renders help and rejects unknown names at one level.

    It deliberately declares no options of its own: everything after the
    command name belongs to the command, and is handed to it untouched.
    """

    parser = argparse.ArgumentParser(
        prog=_program(path),
        description=summary,
        epilog=_nested_commands() if not path else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True
    for node in nodes:
        subparsers.add_parser(node.name, help=node.summary, add_help=False)
    return parser


@contextmanager
def _invoked_as(prog: str, argv: Sequence[str]) -> Iterator[None]:
    """Present the command as if it were its own console script.

    ``argparse.ArgumentParser`` takes its default ``prog`` from
    ``sys.argv[0]``, and the parser is constructed inside the command's own
    ``main``. Presenting ``"glassbox <command>"`` there keeps every command's
    usage line correct without any module knowing it is nested under
    ``glassbox``.
    """

    original = list(sys.argv)
    sys.argv = [prog, *argv]
    try:
        yield
    finally:
        sys.argv = original


def _entry_point(command: Command, path: Sequence[str]) -> Any:
    module_name, _, function_name = command.target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise SystemExit(_unavailable_message(command, path, error)) from error
    return getattr(module, function_name)


def _unavailable_message(
    command: Command, path: Sequence[str], error: ImportError
) -> str:
    program = _program(path)
    if command.extra is None:
        return f"{program} is unavailable: {error}"
    return (
        f"{program} needs the optional '{command.extra}' extra: {error}\n"
        f"Install it and rerun, for example: "
        f"uv run --extra {command.extra} {program} --help"
    )


def _dispatch(
    nodes: Sequence[Node], path: tuple[str, ...], argv: Sequence[str], summary: str
) -> int | None:
    parser = _level_parser(path, nodes, summary)
    head = argv[0] if argv else None
    if (
        head is None
        or head in {"-h", "--help"}
        or head not in {node.name for node in nodes}
    ):
        # argparse renders help for --help (exit 0) and reports a missing or
        # invalid command name on stderr (exit 2); neither call returns.
        parser.parse_args(list(argv[:1]))
        raise AssertionError("unreachable")  # pragma: no cover

    node = next(item for item in nodes if item.name == head)
    rest = list(argv[1:])
    if isinstance(node, Group):
        return _dispatch(node.commands, (*path, node.name), rest, node.summary)
    leaf_path = (*path, node.name)
    with _invoked_as(_program(leaf_path), rest):
        return _entry_point(node, leaf_path)(rest)


def main(argv: Sequence[str] | None = None) -> int | None:
    """Dispatch ``glassbox <command> [...]`` to the command's own front end."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    return _dispatch(TREE, (), arguments, _DESCRIPTION)


if __name__ == "__main__":
    sys.exit(main())
