# Recorded results

Every quantitative claim in the documentation points at a checked-in
artifact under `docs/results/`. This guide covers how those artifacts are
tested, when to regenerate one, and the command that does it.

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

The `ignore` table is where host-dependent values live: wall-clock timing,
platform strings, and anything else that cannot be pinned. It is also where
chaotic derived quantities live, such as counts accumulated over a long
closed-loop simulation whose last-bit numerical differences can flip a
branch taken hundreds of steps earlier. Those counts stay in the artifact,
because they are real recorded evidence, but the test does not assert on
them; asserting on a chaotic count turns ordinary floating-point noise into a
false regression signal.

## When to re-record

Re-record an artifact when a change to the code on its path is an
intentional behavior change: a bug fix, a new feature, a deliberate change to
a formula or a default. The recorded tier is designed to catch the case where
that did not happen, that is, where a refactor was supposed to be
behavior-preserving but the numbers moved anyway.

Re-recording is not a response to the artifact's own provenance metadata
changing on its own. Some artifacts, such as
[`adaptive-recovery-results.json`](../results/adaptive-recovery-results.json),
carry a SHA-256 hash of the source files that produced them. That hash is
provenance, recording what version of the code produced this snapshot; it is
not a trigger. A hash that no longer matches HEAD means the source moved
since the last recording, which is expected between recordings and is not by
itself a reason to re-record.

Never edit a recorded JSON file by hand, and never pick a re-run for its
timings; commit whatever the command wrote.

## The command

```bash
uv run glassbox record-results --list
```

lists every artifact under `docs/results/`: whether it can be regenerated in
this repository, what optional extra and local data it needs, and its
duration class (fast, under about five minutes, or slow).

```bash
uv run glassbox record-results
```

regenerates every fast, regenerable artifact whose requirements are met, and
prints a reason for each one it skips. Add `--include-slow` to also run the
slow ones. `--only NAME [NAME ...]` regenerates specific artifacts by name
and, being an explicit request, runs even a slow one without
`--include-slow`. `--dry-run` prints the exact steps a run would take,
without running them.

Every step is one `glassbox` subcommand, run in-process, exactly as
documented on the artifact's own experiment or concept page; the Cascade X8
artifact additionally has one assembly step with no subcommand form, which
combines several per-step reports into the recorded shape. A handful of
artifacts, mostly built from real-flight or PX4 SITL corpora that are not
checked into the repository, are not regenerable here at all; `--list`
reports those with the reason instead of a duration class.

The command stops at the first failing step and reports which artifact and
which step failed. It never edits documentation prose.

## Doc prose stays hand-written

`glassbox record-results` writes JSON, not prose. After regenerating an
artifact, open the experiment or concept page it belongs to (each is named
in `--list` and in the command's own summary) and update the numbers in the
prose by hand from the new artifact. Superseded numbers are labeled, not
deleted, per the documentation conventions in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md).
