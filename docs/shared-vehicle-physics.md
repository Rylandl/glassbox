# Shared vehicle physics learner

This iteration builds one vehicle learning procedure with shared gravity and
rigid-body integration. It learns six effective body accelerations from observed
motion and arbitrary ordered commands, with causal command filtering and hidden
memory. It uses no vehicle-family branch, mixer, actuator layout or physical
parameter input. Separate fitted weights per configuration are expected.

The named gap is **vehicle mechanics and recursive planning horizon**. The
[frozen protocol](harness/shared-vehicle-physics-v1.json), committed at
`d782c9609711bceb78f720f35a6fe5758ff86c6d`, fixes the architecture, data, optimizer,
four fresh fits, untouched +14M confirmation and separate Dart trial. Complete
implementation and all harness stages must be committed and source-bound before
any scientific fit or simulation.

## Interface

During evaluation the candidate lives in a research module. It is not selected
through a public learner option:

```python
from glassbox.experimental.shared_vehicle import fit, SharedVehicleDynamics

model = fit(recordings)
mean = model.predict(past_states, past_commands, future_commands)
error_half_width = model.envelope()
model.save("vehicle.npz")
restored = SharedVehicleDynamics.load("vehicle.npz")
next_revision = restored.update(new_recordings)
```

`recordings` is a `SequenceCollection`. A telemetry adapter converts available
motion signals to the declared canonical channels: world velocity in NWU,
body angular rate in FLU and a body-to-world rotation matrix. Commands remain
arbitrary ordered channels. The learner does not interpret command names as
motor or control-surface roles. Adapters must establish units, frames and time
alignment from the recording format; they cannot invent missing semantics or
supply vehicle dynamics.

Each forecast consumes 500 ms of observed history and issued commands. It returns
15 motion channels; position can be integrated by the consumer. Shared physics
keeps orientation on SO(3) and applies gravity. A learned acceleration head
represents the configuration's effective response without identifying mass,
inertia or actuator placement. Delayed response is represented by learned
command filters, sampled feature differences and causal hidden memory.

Means can recurse to 1.2 s. Error envelopes remain limited to the fitted 250 ms
horizon, and asking for a longer envelope raises an error. Development recordings
also select the fitted checkpoint, so calibration is not independent. Measured
coverage, physical response derivatives and application adequacy require further
evidence. A rigid-body model is not a claim to represent every articulated or
flexible system.

## Evaluation

Physical comparison uses identical retained Crazyflow/Cascade caches and v4 loss
weights, two new candidate fits and no control refits. Acceptance requires a
joint forecast/response ratio at most 0.97, each kind at most 1.05, and retained
broad regression and targeted angular guards. Absolute errors and missing truth
remain visible. The candidate does not inherit the old arbitrary-channel
synthetic capability verdict.

Dart gets two matched fits from its original eight training and two calibration
recordings, followed by one nominal trial per arm with identical real observed
history, command seed, planner and task. Its historical 1.47 mm result used
simulator-applied thrust at each replan and remains context. The new structured
comparator must infer its latent response from issued commands. Replays,
independent reduction and four artifact alterations must verify evidence before
any acceptance claim.

## Measured result

The fitted implementation is committed at
`431d6a9412546a9066de9511344d5bf188ca9e84`; the final harness correction is
`8cbfd49cd31ae7c5a8170cb9d0fb4aabc2e644cd`. The
[result record](harness/shared-vehicle-physics-v1-result.json) anchors the source,
saved evidence, validation and original failures. Four fixed fits use ordinary recordings:
one shared-physics fit for each simulator, and matched v4/shared-physics fits for
Dart. Physical controls load the retained v4 revisions without refitting. The
new architecture has the same caches, objective weights and 1,000 gradient
attempts as its control; initialization and mechanics change with the architecture.

On the untouched +14M cohort, aggregate forecast error falls **43.95%**, command
response error falls **22.12%**, and their equally weighted joint ratio is
**0.66071**. These are geometric means across the prospectively weighted physical
signal groups, horizons and conditions, not percentages of predictions that are
accurate. The frozen residual verdict is **failed**: Cascade's wind forecast and
response ratios are **3.73823** and **1.64347**, exceeding the 1.5 scope limit.
Its primary rotation-response parent-p95 ratio is **1.52989**, exceeding the
1.5 tail limit. All four targeted Crazyflow angular retention checks pass.
The gains support continuing this shared-physics approach; the failed checks
remain failed and the public recipe remains v4.

Primary 250 ms component RMSE, retained v4 → shared physics:

| Simulator | Forecast velocity, m/s | Forecast rate, rad/s | Response velocity, m/s | Response rate, rad/s |
| --- | ---: | ---: | ---: | ---: |
| Crazyflow | 0.03574 → 0.00916 | 0.12267 → 0.06890 | 0.01894 → 0.01217 | 0.12872 → 0.09839 |
| Cascade | 0.06459 → 0.04723 | 0.04938 → 0.03465 | 0.02106 → 0.01793 | 0.01950 → 0.02136 |

Crazyflow completes 66/84 parents, with 18 altitude failures. Primary 250 ms truth
is available for 444/480 forecast and 712/768 response queries. Cascade completes
84/84 parents, with 480/480 forecast and 576/576 response queries at that horizon.
Both arms use exactly the same available truth; incomplete collection remains
visible and does not count as a model success.

Both physical replays reproduce native truth, saved checkpoints, calibration and
all predictions exactly. An independent reduction reproduces 257,040 error rows;
all four altered-artifact challenges are rejected. This verifies the failed
scientific verdict as well as the measured gains.

## Dart evidence

Both fits consume the original eight three-second training recordings and two
development recordings through identical 1,536/256-window caches. Those windows
overlap; they are not 1,536 independent experiments. V4 rejects all 1,000 permitted
Adam proposals and stays at its ridge initialization. Even its smallest proposal
raises training loss 6.02%. This reveals an optimizer limitation rather than
establishing that the old representation exhausted its capacity.

The shared model selects step 400, with matched weighted development loss
0.00925356 versus v4's 0.10027121. On the untouched test-10/test-11 recordings,
all 72 queries per model have finite forecasts and command JVPs. Descriptive
endpoint component RMSE pools queries equally:

| Horizon | Velocity v4 → shared, m/s | Rate v4 → shared, rad/s |
| --- | ---: | ---: |
| 50 ms | 0.02659 → 0.01075 | 0.07083 → 0.05120 |
| 150 ms | 0.19948 → 0.03420 | 0.43301 → 0.33461 |
| 250 ms | 0.64790 → 0.08772 | 1.02180 → 0.83528 |
| 1.2 s | 31.57728 → 4.36472 | 19.69728 → 10.29175 |

There are 20 queries per short horizon and 12 at 1.2 s. The improvements are
substantial, but the absolute long-horizon errors remain large. Computationally
finite JVPs do not establish physical derivative accuracy. These diagnostics do
not replace the separately declared closed-loop contact task.

The completed nominal contact trials use the unchanged Dart planner, original
command seed, a common 500 ms observed prelude and a 1.2 s planning horizon.
The structured comparator reconstructs its actuator state from issued commands;
no arm receives simulator actuator state. All arms receive ideal physical motion
observations. These are simulator trials, with no real-time performance claim.

| Model | Executed flight | Contact task | Completed/converged solves |
| --- | ---: | --- | ---: |
| Shared vehicle | 0.81 s | Failed: no plane crossing; planning stopped on a nonfinite initial objective or gradient | 27 / 1 |
| V4 research extrapolation | 0 s | Failed before executing commands, with the same error | 0 / 0 |
| Structured causal history | 1.20 s | Passed: 8.17 mm miss, 0.727° axis error | 40 / 38 |

The structured contact occurs at 1.19574 s with normal/tangent speeds
1.40463/0.05262 m/s; all four task checks pass. The shared model's closest approach
to the contact plane is 0.34585 m. Its 27 nonfinite objective/gradient evaluations
and only one converged solve make clear that executing part of the flight is not
evidence of successful learned control. There is one nominal trial per arm, not
a measured success rate.

Two separately committed harness corrections preserve the original failures. An
obsolete import stopped the first launch before any plant/controller activity.
The corrected launch completed the first shared solve, then failed serializing a
nonfinite diagnostic. The final correction authenticated and resumed that saved
solve without repeating it or its prelude. Strict scientific JSON remains strict;
nonfinite optimizer diagnostics are recorded explicitly. No completed fit or
optimizer solve was repeated. Both Dart fit replays, the forecast replay and the
final composite control replay pass exactly; control replay performs no optimization.

## Remaining model gap

The next named gap is **Dart task-horizon command-response fidelity**. Training
uses 250 ms losses on 24 seconds of motion; planning recurses for 1.2 s. In the
saved initial plan, 33/120 command rows exceed at least one training channel's
observed range, and channel differences are substantially larger than in training.
The saved rollout leaves the observed angular-rate ranges after 0.58 s even
while its orientation remains valid. Shared geometry has improved prediction,
but the learned acceleration field still needs accurate, well-conditioned
responses across the task's command and time range.

This is a diagnosis of a measured gap, not proof of the precise nonfinite trigger:
the failing optimizer proposal's commands were not saved. The next iteration
must freeze one targeted remedy and direct long-horizon response measurements,
while retaining the crosswind and response-tail deficits as regression obligations.
No next experiment has been started. The research implementation is retained for
that work; it is not a qualified replacement for the public recipe.
