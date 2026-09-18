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

## Measured response and its limits

The frozen experiment at `be994c9` preserves the original 37-completed-case
cohort and all 189 of its command channels at every horizon. Cumulative detected
activity is:

| Response horizon | As shipped | Engine startup | Matched gains / losses |
| --- | ---: | ---: | ---: |
| 0.25 s | 45/189 | 115/189 | 70 / 0 |
| 1 s | 47/189 | 123/189 | 76 / 0 |
| 2 s | 48/189 | 125/189 | 77 / 0 |
| 5 s | 48/189 | 125/189 | 77 / 0 |

These counts use small detectability thresholds, not useful control-authority
or prediction-accuracy tolerances. Fixed endpoint activity at 5 seconds is
47/189 and 122/189, respectively. Among the 78 throttles, cumulative activity
rises from 1 to 35 at 0.25 seconds and from 3 to 45 at 5 seconds. No learner was
fitted or scored, and no controller ran.

Both arms retain 37 completed configurations, 25 missing initializations, three
load failures and one parent numerical failure. `minisgs` contributes three
additional detectable 0.25-second channels, then fails before the longer
horizons. Thus the all-candidate short-horizon counts are 48/192 and 118/192;
using these as if they were the old 189-channel cohort would change the
comparison. Its finite prefix is not admitted as a useful learning regime.

The traces separate three practical causes. B17's shipped engines stay at zero
running flag, RPM and thrust; startup supplies thrust and detectable throttle
response. Global5000 already runs its engines in both arms, but the command
change takes about 0.9 seconds to produce a measured thrust and motion response.
F450's indexed throttle commands change without changing the corresponding
motor positions, thrust or motion through 5 seconds; its collective command
works. Its electric engines also deliver power while the running flag is false.
These are specific observations under the frozen inputs, not universal engine
or controllability claims.

Finite completion still permits invalid conditions: L410 passes below ground
in both arms and reaches about 507 m/s after startup; its reported thrust is
also suspect. Startup snapshots alone can mislead: F450's immediate post-startup
thrust increase does not persist into a different subsequent trajectory.
The next iteration must declare and verify useful operating conditions and
actual effective command mappings before generating a learning corpus.

All 132 outcomes and 34,716 arrays reproduce exactly in fresh execution. The
independent reductions agree, all eight alteration tests are rejected, and
146 focused tests plus Ruff pass. The [result record](harness/jsbsim-excitation-v1-result.json)
pins the 403-file evidence bundle, full replay, separate reductions and
adversarial audit. Public learner, controller and consumer interfaces are unchanged.
