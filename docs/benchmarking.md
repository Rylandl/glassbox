# Fast readout benchmark loop

Use one runner for the known online readout recordings. A candidate supplies a
factory `module:Class`; `Class(prefix)` creates a fresh episode fit, exposes
`session.model` and `session.predict(past, past_commands, future_commands)`,
and implements `observe(row, command, next_state)`. The runner owns source
loading, causal update order, origins, scoring, controls, timing, artifact
format and the compact table. A model change should no longer copy an
evaluator.

The default command runs the complete known-recording suite: all eight cases,
4,187 causal updates and 263 previously frozen forecast origins through the
end of each stream. This includes late flight conditions and the quad
configuration change. The table keeps first-origin error visible beside
all-origin 250 ms velocity/rate RMSE, worst body-rate error and update time.
All five physical horizons, orientation and native one-step scores remain in
the saved summary. Candidate, prior direct readout and historical full learner
are shown side by side; full-learner one-step scores are not invented when that
control is absent. The optional four-case smoke screen still uses two early
origins to find obvious failures quickly.

```sh
./.venv/bin/python scripts/benchmark_online_readout.py run \
  --candidate candidate_module:Candidate \
  --output artifacts/example-full

./.venv/bin/python scripts/benchmark_online_readout.py verify \
  --output artifacts/example-full \
  --manifest-sha256 HASH_PRINTED_BY_RUN
```

The runner writes `table.md`, `summary.json`, saved predictions/timings and a
sealed manifest. Verification re-authenticates the recorded inputs and
controls and recomputes scores from saved predictions without fitting. It
records candidate source and runner hashes, so an exploratory smoke can be
traced. Commit the candidate before using a full result for an engineering
decision. A separate candidate-specific unit test is warranted only for a
new derivative, causality or numerical claim.

The original two-origin runner was validated against the archived physical
SO(3) direct readout. The expanded full-stream runner has its own [validation
index](online-readout-benchmark-validation.json); its forecasts and one-step
streams are checked against the same archived evaluation.

This benchmark uses known recordings and a fresh but nonzero causal prefix;
it is not held-out-vehicle, counterfactual command-response or live Throw
evidence. The runner itself has no model-family branch. Broader qualification
remains a separate step after a fast model idea survives this screen.
