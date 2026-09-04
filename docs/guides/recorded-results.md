# Recorded results

Every quantitative claim on an experiment or concept page is the literal
content of a checked-in artifact under `docs/results/`. Each artifact is
machine output, produced by one command over one manifest entry, and pinned by
one test. This guide is the table of those artifacts, the tiers they belong
to, and the commands that regenerate them. The literature review is the
exception: its negative results are recorded as prose in the decision record,
not as machine-readable artifacts.

## The eight artifacts

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

The five `validation-*` artifacts are the corpus tier's first recording and
are not committed yet; `--list` marks them pending, and the manifest already
carries the exact chain and contract that will produce them.

## Two tiers

The **local** tier runs in this repository with nothing downloaded. It is what
continuous integration checks on every pull request, with

```bash
uv run glassbox record-results --check --tier local
```

which regenerates both local artifacts into a temporary directory and compares
each against the committed file, ignoring only the paths that entry declares
volatile: the environment block, the source fingerprint, and per-trace wall
clock. Any other difference fails the job, so a change that was meant to be
behavior-preserving cannot move a recorded number unnoticed.

The **corpus** tier is a maintainer job, documented in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md). It needs the pinned public corpora
on disk, the `px4`, `ros` and `cascade` extras, and hours of fitting, so it
does not run in continuous integration. Run it with

```bash
uv run glassbox record-results --tier corpus
```

## What a corpus validation artifact contains

The five corpus artifacts share one chain, `corpus prepare` then `fit` then
`evaluate`, and one contract:

- `corpus`: the registry's citation, license, pinned version and the SHA-256
  digest of every pinned file the run read. A corpus that does not verify is a
  different corpus, and a number measured on it is not the published one.
- `protocol`: the named scoring policy the evaluation ran under, with its
  stride, baseline, floors and score reduction. Two numbers produced under
  different conventions are not comparable, so the whole policy is recorded.
- `fit`: one block per scored model, or one per fold for a leave-one-out run,
  with the fit specification, the split, the optimization record and the
  parameter evidence the fit produced.
- `results` and `baseline`: the evaluation report's aggregate and per-flight
  numbers, and what they were scored against.
- `implementation` and `environment`: the SHA-256 of the source modules a
  validation number passes through, and the host it ran on.
- `smoke`: false on a recording. A smoke run sets it true.

## Exercising the chain without recording

```bash
uv run glassbox record-results --tier corpus --smoke /tmp/glassbox-smoke \
  --source-root artifacts
```

runs every corpus chain on a two-step fit budget and two folds, writing every
prepared trajectory, report and artifact under the smoke directory instead of
into the repository, and reusing corpora already verified under
`--source-root` rather than downloading them again. Every artifact it produces
carries `"smoke": true`, so smoke output can never be read as evidence.
`--fit-steps N` and `--limit-folds N` change the shortened budgets.

## Two tiers of recorded-result tests

Each recorded artifact has a test with two tiers, not one.

The contract tier asserts the claims the documentation actually makes and
that survive floating-point noise: that a report has the expected shape, that
its semantics flags say what the prose says they say, that specific
comparisons hold in the direction claimed. This tier does not compare against
the checked-in JSON at all, so it stays meaningful even when the recorded
numbers move.

The recorded tier compares a fresh run against the checked-in artifact, field
by field, under a tolerance chosen per quantity and never tighter than that
quantity's own sensitivity. `tests/_recorded.py` implements this as
`assert_recorded_close`, driven by three pattern tables next to each test:
`tolerances` (a relative tolerance, or a relative/absolute pair, per path
pattern), `exact` (paths that must compare equal, for values that are
genuinely deterministic offline), and `ignore` (paths excluded from
comparison entirely). A float that matches no pattern is a test failure
rather than a silent pass, so the table cannot quietly stop covering part of
an artifact.

The `ignore` table is the manifest entry's own `volatile` table, so the test
and `record-results --check` exclude exactly the same paths. That is where
host-dependent values live: wall-clock timing, platform strings, and the
source fingerprint. It is also where chaotic derived quantities live, such as
counts accumulated over a long closed-loop simulation whose last-bit numerical
differences can flip a branch taken hundreds of steps earlier. Those counts
stay in the artifact, because they are real recorded evidence, but the test
does not assert on them; asserting on a chaotic count turns ordinary
floating-point noise into a false regression signal.

## When to re-record

Re-record an artifact when a change to the code on its path is an
intentional behavior change: a bug fix, a new feature, a deliberate change to
a formula or a default. The recorded tier and `--check` are designed to catch
the case where that did not happen, that is, where a refactor was supposed to
be behavior-preserving but the numbers moved anyway.

Re-recording is not a response to the artifact's own provenance metadata
changing on its own. The adaptive-recovery artifact and every corpus
validation artifact carry a SHA-256 hash of the source files that produced
them. That hash is provenance, recording which version of the code produced
that snapshot; it is not a trigger. A hash that no longer matches HEAD means
the source moved since the last recording, which is expected between
recordings and is not by itself a reason to re-record.

Never edit a recorded JSON file by hand, and never pick a re-run for its
timings; commit whatever the command wrote.

## The command

```bash
uv run glassbox record-results --list
```

lists every artifact under `docs/results/`: its tier, its inputs, and whether
it is recorded, pending its first run, or blocked by a missing extra or
missing local data.

```bash
uv run glassbox record-results
```

regenerates every artifact in the local tier whose requirements are met, and
prints a reason for each one it skips. `--only NAME [NAME ...]` regenerates
specific artifacts by name and, being an explicit request, ignores the tier
gate. `--dry-run` prints the exact steps a run would take, without running
them; with neither `--only` nor `--tier` it prints the plan for every artifact
in the manifest.

Every step is one `glassbox` subcommand, run in-process, exactly as documented
on the artifact's own experiment or concept page; each corpus artifact and the
Cascade X8 artifact additionally end in one assembly step with no subcommand
form, which combines the chain's own reports into the recorded shape.

The command stops at the first failing step and reports which artifact and
which step failed. It never edits documentation prose.

## Doc prose stays hand-written

`glassbox record-results` writes JSON, not prose. After regenerating an
artifact, open the experiment or concept page it belongs to (each is named
in `--list` and in the command's own summary) and update the numbers in the
prose by hand from the new artifact. Superseded numbers are labeled, not
deleted, per the documentation conventions in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md).
