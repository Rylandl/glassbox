# Fast readout benchmark loop

Use one runner for the known online readout recordings. A candidate supplies a
factory `module:Class`; `Class(prefix)` creates a fresh episode fit, exposes
`session.model` and `session.predict(past, past_commands, future_commands)`,
and implements `observe(row, command, next_state)`. The runner owns source
loading, causal update order, origins, scoring, controls, timing, artifact
format and the compact table. A model change should no longer copy an
evaluator.

Run the four-case smoke screen first: both fixed-wing recordings, the hard
quad and a gentle paired quad. It is meant to find obvious failures quickly.
Run all eight cases only when that screen is promising. Both suites use the
same [frozen source roster](online-readout-benchmark.json), identical first
and next conditional origins, and physical 50–250 ms scores. The table shows
candidate, prior direct readout and historical full learner side by side;
full-learner one-step scores are not invented when that control is absent.

```sh
./.venv/bin/python scripts/benchmark_online_readout.py run \
  --candidate candidate_module:Candidate \
  --suite smoke \
  --output artifacts/example-smoke

./.venv/bin/python scripts/benchmark_online_readout.py verify \
  --output artifacts/example-smoke \
  --manifest-sha256 HASH_PRINTED_BY_RUN
```

The runner writes `table.md`, `summary.json`, saved predictions/timings and a
sealed manifest. Verification re-authenticates the recorded inputs and
controls and recomputes scores from saved predictions without fitting. It
records candidate source and runner hashes, so an exploratory smoke can be
traced. Commit the candidate before using a full result for an engineering
decision. A separate candidate-specific unit test is warranted only for a
new derivative, causality or numerical claim.

This benchmark uses known recordings and a fresh but nonzero causal prefix;
it is not held-out-vehicle, full-stream or live Throw evidence. The runner
itself has no model-family branch. Broader qualification remains a separate
step after a fast model idea survives this screen.
