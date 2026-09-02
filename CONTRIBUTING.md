# Contributing

## Setup

```bash
uv sync --dev --extra crazyflow --extra crazyflow-animation --extra cascade
uv run pre-commit install   # optional; mirrors CI locally
```

`uv sync` makes the environment exact, so a later `uv sync --dev` without the
extras removes the simulators and their tests skip. Pass the extras every time,
or the simulator-backed tests silently stop running.

The default suite runs in about a quarter of an hour:

```bash
uv run pytest
```

Tests marked `crazyflow` and `cascade` need those extras and skip cleanly
without them. The PX4 SITL contract tests are opt-in with
`GLASSBOX_RUN_PX4_SITL=1`.

## Checks

CI runs `ruff check`, `ruff format --check`, and the default test suite on every
push and pull request. Run the same locally with:

```bash
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
```

## Recorded results

Several benchmarks pin a recorded artifact under `docs/results/` and compare a
fresh run against it. See [the recorded-results guide](docs/guides/recorded-results.md)
for the two-tier test policy and when to re-record: in short, re-record after
an intentional behavior change on that artifact's path, not in response to its
own provenance metadata changing on its own.

`glassbox adaptive-recovery` also records a SHA-256 hash of the source files
that produced its artifact (`adaptive_recovery_benchmark.BENCHMARK_SOURCE_FILES`).
That hash is provenance, not a trigger; it is expected to drift between
recordings and does not by itself require a re-record.

Regenerate a recorded artifact with:

```bash
uv run glassbox record-results --only <artifact-name>
```

or see `uv run glassbox record-results --list` for every artifact's name and
status. Commit the JSON alongside the change that motivated it. Never edit a
recorded JSON by hand, and never pick a re-run for its timings.

## Documentation conventions

- Every quantitative claim points at a recorded artifact or a reproducible
  command. Superseded numbers are labeled, not deleted.
- Do not quote absolute wall-clock timings in prose; they depend on the host.
  Ratios and bounded statements are fine. Timing fields in artifacts are
  marked nondeterministic.
- Experiment pages follow one layout: what this establishes, purpose, data,
  reproduce, results, boundary.
- Plain prose, no em-dashes.

## Layout

`import glassbox` must load only `glassbox.core`, `glassbox.belief`, and
`glassbox.control.nmpc`; the `workflows`, `io`, `cli`, `integrations`, and
`experimental` subpackages import on demand. `tests/test_public_api.py` guards
this and pins the exported names.
