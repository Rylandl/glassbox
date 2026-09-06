# Contributing

## Setup

```bash
uv sync --dev --extra cascade
uv run pre-commit install   # optional; mirrors CI locally
```

`uv sync` makes the environment exact, so a later `uv sync --dev` without the
extra removes the simulator and its tests skip. Pass the extra every time, or
the simulator-backed tests silently stop running.

## Tests

```bash
uv run pytest
```

The default suite takes several minutes. `-m "not slow"` skips the
benchmark-scale tests. Tests marked `cascade` need that extra and skip cleanly
without it; deselect them with `-m "not cascade"`. The PX4 SITL contract tests
are opt-in and need Docker:

```bash
GLASSBOX_RUN_PX4_SITL=1 uv run pytest -m px4_sitl tests/integration/test_px4_sitl.py -v
```

To exercise all four airborne shadow profiles, also set
`GLASSBOX_RUN_PX4_FLIGHT_SHADOW=1` and point `GLASSBOX_PX4_NMPC_MODEL` at an
actionable quadrotor artifact. The fixture flies a disposable simulator through
the maintained CLI, prewarms the shadow controller, and waits for the first
excitation target before checking the recorded motion. Per-profile JSON traces
and summaries are written under pytest's temporary directory. The separate
fixed-command test instead requires `GLASSBOX_PX4_NMPC_COMMAND` and leaves
`GLASSBOX_RUN_PX4_FLIGHT_SHADOW` unset.

The offline recovery investigation is reproduced with
`uv run python scripts/investigate_recovery.py --output docs/investigations/recovery.json`.
It is a controlled diagnostic with an independent optimizer reference, outside
the recorded-results manifest; see [the investigation](docs/recovery-investigation.md)
for its uncertainty ablations and limits.

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
command produces or checks each of them. Every quantitative claim in
[`docs/validation.md`](docs/validation.md) and on the concept pages is the
literal content of one of these files, named there by its path and its key.

| Artifact | Tier | Inputs | Regenerate with |
| --- | --- | --- | --- |
| `adaptive-recovery-results.json` | local | synthetic scenarios generated in-process | `glassbox record-results --only adaptive-recovery-results` |
| `nmpc-acceptance-results.json` | local | synthetic scenarios generated in-process | `glassbox record-results --only nmpc-acceptance-results` |
| `validation-nanodrone-results.json` | corpus | pinned `nanodrone` corpus | `glassbox record-results --only validation-nanodrone-results` |
| `validation-arp-results.json` | corpus | pinned `arp` corpus, `px4` extra | `glassbox record-results --only validation-arp-results` |
| `validation-idf-results.json` | corpus | pinned `idf` corpus, `px4` extra | `glassbox record-results --only validation-idf-results` |
| `validation-x8-results.json` | corpus | pinned `x8` corpus | `glassbox record-results --only validation-x8-results` |
| `validation-epfl-results.json` | corpus | pinned `epfl` corpus, `ros` extra | `glassbox record-results --only validation-epfl-results` |
| `cascade-x8-validation-results.json` | corpus | pinned `x8` corpus, `cascade` extra | `glassbox record-results --only cascade-x8-validation-results` |

`glassbox record-results --list` prints the same table with each artifact's
current status: recorded, pending its first run, or blocked by a missing extra
or missing local data. `--dry-run` prints the exact steps a run would take.
Every step is one `glassbox` subcommand run in-process, and each corpus
artifact ends in one assembly step that combines the chain's own reports into
the recorded shape. Recording disables holdout resume so a cached fit from
earlier code cannot be presented as a freshly regenerated result.

The **local** tier runs in this repository with nothing downloaded, and is
what CI checks:

```bash
uv run glassbox record-results --check --tier local
```

That regenerates both local artifacts into a temporary directory and compares
each against the committed file, ignoring only the paths that entry declares
volatile: the environment block, the source fingerprint and per-trace wall
clock. Any other difference fails the job, so a change meant to be
behavior-preserving cannot move a recorded number unnoticed.

### The corpus tier is a maintainer job

The corpus tier is not run in CI. It needs

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
nothing except that the chain runs. `--fit-steps N` and `--limit-folds N`
change the shortened budgets.

The PX4 SITL benchmarks are not in the manifest at all: recording one needs a
running SITL container, so their numbers are prose on
[`docs/validation.md`](docs/validation.md) and say so there.

The offline recovery investigations also live outside the manifest. Earlier
reports retain their recorded revisions as historical evidence. Rerun the
supervised cases against the current code with:

```bash
uv run python scripts/investigate_supervised_recovery.py \
  --output /tmp/glassbox-supervised-recovery.json
```

This drives the production loop against a synthetic plant with a discrete
supervision clock and deliberately injected faults. Its nominal solver deadline
is disabled, so it measures command selection and recovery rather than real-time
readiness. The [investigation](docs/recovery-investigation.md#supervised-recovery-and-model-support)
describes the calibration/validation split and retained negative results.

To compare NMPC alone under soft and explicit model-support constraints:

```bash
uv run python scripts/investigate_constrained_recovery.py \
  --output /tmp/glassbox-constrained-recovery.json
```

This reference keeps the original envelope and uncertainty. A failed solve ends
its scenario without applying a fallback. Its SLSQP feasibility and Lagrangian
residuals are separate from the production solver's convergence flag; it is an
offline formulation check and is not wired into the control loop.

The first SQP investigation (snapshot `8a79306`) includes the former clipping
and warm-start rules as explicit ablations. Rerun it with:

```bash
uv run python scripts/investigate_sqp_recovery.py \
  --output /tmp/glassbox-sqp-recovery.json
```

The current runtime follow-up measures seed reuse, the explicit model constraint
interface, and deadline enforcement (including a separately declared startup
budget):

```bash
uv run python scripts/investigate_sqp_runtime.py \
  --output docs/investigations/sqp-runtime.json
```

Run timing comparisons without other benchmark or test processes. Both
prewarm each controller and record complete solve times, separating the cold
optimization from subsequent solves. The first experiment disables deadlines;
the runtime follow-up records each case's startup and subsequent deadline
budgets and stops immediately when an unusable solve is returned.

### When to re-record

Re-record an artifact when a change to the code on its path is an intentional
behavior change: a bug fix, a new feature, a deliberate change to a formula or
a default. **A change that moves a recorded number re-records that artifact in
the same commit, and the commit message says which numbers moved and why.**
The recorded tier and `--check` exist to catch the other case, where a refactor
was supposed to be behavior-preserving but the numbers moved anyway.

Re-recording is not a response to provenance metadata changing on its own. The
adaptive-recovery artifact and every corpus validation artifact record a
SHA-256 hash of the source files that produced them. That hash is provenance,
not a trigger; it is expected to drift between recordings, which is why those
manifest entries list it as volatile and neither the pinned tests nor
`record-results --check` compares it.

Never edit a recorded JSON by hand, and never pick a re-run for its timings.
Commit whatever the command wrote, alongside the change that motivated it.

`glassbox record-results` writes JSON, not prose. After regenerating an
artifact, open the page that cites it and update the numbers by hand from the
new file.

### The two tiers of recorded-result tests

Each recorded artifact has a test with two tiers. The **contract** tier
asserts the claims the documentation actually makes and that survive
floating-point noise: that a report has the expected shape, that its semantics
flags say what the prose says they say, that specific comparisons hold in the
direction claimed. It does not compare against the checked-in JSON at all, so
it stays meaningful even when the recorded numbers move.

The **recorded** tier compares a fresh run against the checked-in artifact,
field by field, under a tolerance chosen per quantity and never tighter than
that quantity's own sensitivity. `tests/_recorded.py` implements this as
`assert_recorded_close`, driven by three pattern tables next to each test:
`tolerances`, `exact` and `ignore`. A float that matches no pattern is a test
failure rather than a silent pass, so the table cannot quietly stop covering
part of an artifact. The `ignore` table is the manifest entry's own `volatile`
table, so the test and `record-results --check` exclude exactly the same
paths: wall-clock timing, platform strings, the source fingerprint, and
chaotic derived quantities such as counts accumulated over a long closed-loop
simulation whose last-bit differences can flip a branch taken hundreds of
steps earlier. Those counts stay in the artifact because they are real
recorded evidence, but asserting on them would turn ordinary floating-point
noise into a false regression signal.

## Documentation conventions

- The documentation is eight pages. Adding a ninth needs a reason; folding a
  section into an existing page is the default.
- Every quantitative claim on `docs/validation.md` or a concept page is the
  literal content of a recorded artifact, cited by its path and its key so a
  reader can check it. A claim with no artifact is not published as a number.
- A recorded number is written down once. `docs/validation.md` is where it
  lives; a concept page describes what the measurement establishes and links
  to the section, so a re-record moves one page rather than two. Each manifest
  entry's `doc_page` names the section that has to be updated.
- Negative results and withdrawn approaches are recorded as prose in
  [`docs/literature-review.md`](docs/literature-review.md) with the last commit
  that carried their code. Their code and their artifacts are not kept.
- Do not quote absolute wall-clock timings in prose; they depend on the host.
  Ratios and bounded statements are fine. Timing fields in artifacts are
  marked volatile.
- Every command shown in a page must resolve against the current CLI tree, and
  every relative link must resolve.
- Every Python snippet on the README or a concept page runs. A page's snippets
  are read in order, so a later one may use names an earlier one defined.
- Plain prose, no em-dashes.

## Layout

`import glassbox` must load only `glassbox.core`, `glassbox.belief` and
`glassbox.control`; the `workflows`, `io`, `cli` and `integrations`
subpackages import on demand. `tests/test_public_api.py` guards this and pins
the exported names.
