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

Status: implementation complete and regression checks pass. The four frozen
fits and physical/Dart evaluations are next. Public v4 remains the adopted
baseline until the successor is qualified.
