# Contributing

## Setup

```bash
uv sync --dev
uv run pre-commit install   # optional; mirrors CI locally
```

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
fresh run against it, and `glassbox-adaptive-recovery` also hashes its source
files into that artifact. After changing `belief.py`, `adaptation.py`,
`dynamics.py`, `evaluation.py`, the NMPC package, or the other files listed in
`adaptive_recovery_benchmark.BENCHMARK_SOURCE_FILES`, re-record with the
command on the experiment page and commit the JSON alongside the change. Never
edit a recorded JSON by hand, and never pick a re-run for its timings.

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

`import glassbox` must load only the core; workflow, corpus, CLI, and
integration modules import on demand. `tests/test_public_api.py` guards this.
