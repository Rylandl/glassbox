# JSBSim startup and response-timescale diagnosis

This is a no-fit experiment about the informativeness of the proposed benchmark,
not a qualification of the learner or a demonstration of flight. The frozen
[protocol](harness/jsbsim-excitation-v1.json) retains all 66 configurations from
the preceding [onboarding audit](jsbsim-onboarding.md), their selected shipped
initializations, command contracts and assets.

Two arms receive the same extended deterministic command tape. `as_shipped`
uses the original initialization. `engine_bootstrap` additionally calls JSBSim's
`get_propulsion().init_running(-1)` once immediately after successful `run_ic`.
This is an engine startup package, including internal throttle/mixture and
steady-state calculations; it does not isolate a running flag. No trim, additional
integration, command restoration, control-mode repair or controller is introduced.

Each parent and replay spans 120 intervals at 20 Hz. Fresh factual and signed
command branches replay the first 20 intervals, then sustain the prescribed
intervention for 100 intervals. Responses are evaluated after 0.25, 1, 2 and
5 seconds. A late numerical failure preserves earlier complete horizon results.
Timeouts or native crashes leave the entire affected arm unavailable, with
checkpointed prefixes retained only as descriptive evidence. Missing observations
never count as weak response.

The engine flag is read from the catalogued `set-running` property. Thrust,
spool/rotor speed, fuel flow, effective actuator positions and ground context are
read where present. Absent properties, getter errors and nonfinite diagnostics
are recorded explicitly. These values are diagnostics, not additional physical
validity gates, and a running flag is not a universal measure of delivered power.

Cumulative threshold detection and fixed endpoint response magnitudes are
reported separately. A longer opportunity to cross a threshold is not stronger
intrinsic control authority. Every horizon includes the active/weak/unavailable
transition table between arms, matched available gains and losses, and the
original 37-completed-configuration cohort alongside all newly available cases.
A finite but exploding trajectory remains visible and is not a usable operating
envelope merely because its response is detectable.

## Reproduction

Use the pinned JSBSim 1.3.1 runtime and release data, and the externally sealed
previous onboarding evidence bundle. `DATA` and `REFERENCE` below denote their
local directories. This experiment pins its own source inventory; historical
onboarding replay remains on its original implementation checkout.

```sh
python -m glassbox.experimental.jsbsim_excitation run \
  --root "$DATA" --reference "$REFERENCE" --out "$OUTPUT" --workers 4
python -m glassbox.experimental.jsbsim_excitation verify \
  --root "$DATA" --reference "$REFERENCE" --out "$OUTPUT" --workers 4 \
  --seal-sha256 "$PUBLISHED_RUN_SHA256"
```

`validate` authenticates reference bytes, runtime/source/asset identities and
recomputes saved evidence. `verify` also regenerates every trajectory in fresh
worker processes and compares dtype, shape and bytes, including telemetry and
failed observations. It requires the externally published run digest; an
internally consistent rewritten physics trajectory does not authenticate itself.
Each candidate-arm receives its own 180-second budget. Positive Python errors
abort the experiment instead of being counted as aircraft setup failures.
