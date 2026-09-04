# Contributing

## Setup

```bash
uv sync --dev --extra cascade
uv run pre-commit install   # optional; mirrors CI locally
```

`uv sync` makes the environment exact, so a later `uv sync --dev` without the
extra removes the simulator and its tests skip. Pass the extra every time, or
the simulator-backed tests silently stop running.

The default suite takes several minutes; `-m "not slow"` skips the three
benchmark-scale tests:

```bash
uv run pytest
```

Tests marked `cascade` need that extra and skip cleanly without it. The PX4
SITL contract tests are opt-in with `GLASSBOX_RUN_PX4_SITL=1`.

## Checks

CI runs `ruff check`, `ruff format --check`, the default test suite, and the
local tier of the recorded-results manifest on every push and pull request.
Run the same locally with:

```bash
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
uv run glassbox record-results --check --tier local
```

## Recorded results

One manifest names every artifact under `docs/results/`, in two tiers, and one
command produces or checks each of them. See
[the recorded-results guide](docs/guides/recorded-results.md) for the table,
the two-tier test policy and when to re-record: in short, re-record after an
intentional behavior change on that artifact's path, not in response to its own
provenance metadata changing on its own.

The adaptive-recovery artifact and every corpus validation artifact record a
SHA-256 hash of the source files that produced them. That hash is provenance,
not a trigger; it is expected to drift between recordings and does not by
itself require a re-record, which is why their manifest entries list it as
volatile and neither the pinned tests nor `record-results --check` compares
it.

Regenerate a recorded artifact with:

```bash
uv run glassbox record-results --only <artifact-name>
```

or see `uv run glassbox record-results --list` for every artifact's name and
status. Commit the JSON alongside the change that motivated it. Never edit a
recorded JSON by hand, and never pick a re-run for its timings.

### The corpus tier is a maintainer job

`record-results --check --tier local` is what CI runs: two synthetic-scenario
artifacts and no download. The corpus tier is not run in CI. It needs

- the five pinned public corpora fetched and verified on disk, which is
  several gigabytes and includes one 2.12 GB archive,
- the `px4`, `ros` and `cascade` extras, the last of which is a Git dependency
  on the Cascade fixed-wing simulator,
- and hours of fitting: thirteen leave-one-session-out folds on the IDF-DS
  corpus alone.

Run it deliberately, on a machine that has the corpora:

```bash
uv run --all-extras glassbox record-results --tier corpus
```

Before running it for real, exercise the chain on a tiny budget, which touches
nothing in the repository:

```bash
uv run --all-extras glassbox record-results --tier corpus \
  --smoke /tmp/glassbox-smoke --source-root artifacts
```

Every artifact a smoke run produces carries `"smoke": true` and is evidence of
nothing except that the chain runs.

The PX4 SITL benchmarks are not in the manifest at all: recording one needs a
running SITL container, so their pages carry their numbers as prose and say so.

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
