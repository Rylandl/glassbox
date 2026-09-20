# Run Dart with a general vehicle model

The user wants one learner across vehicle types and configurations, with shared
physical knowledge and no catalog. Dart's precise pose/contact task is the
immediate product milestone. This plan records that direction; it freezes no
experiment, changes no fitted model and claims no new task result.

## Established differences

The current generic rollout in `src/glassbox/_sequence_model.py` learns additive
changes to every observed channel. The vehicle adapter uses world velocity,
body rates and nine rotation-matrix entries. The recurrence has quadratic terms
and learned memory, but does not enforce valid rotations, integrate attitude
from angular velocity or explicitly separate gravity and body acceleration.
The predictor is tied to the fitted sampling grid.

The structured path in `src/glassbox/core/dynamics.py` supplies position/attitude
kinematics, coordinate transforms, gravity, normalized quaternion integration
and explicit latent actuator response. It also supplies vehicle-family force
laws and mixer assumptions. We should preserve useful common mechanics while
learning the configuration-dependent response through one shared formulation.
The existing structured path is a comparator, not that proposed generic learner.

Dart's [documented results](/Users/ryland/autonomy/dart/docs/results.md) include
1.47 mm contact error and 0.30 degree top-axis error with the structured mean
and terminal MPC. Its learned open-loop plan achieved 20.10 mm and 3.37 degrees.
The recordings and modeling assumptions together support useful dynamics in
that setting. This weakens treating more data as the default explanation for
generic-model failures. Feedback and physical assumptions still contribute to
the demonstrated result.

The [saved-data comparison](/Users/ryland/autonomy/dart/docs/glassbox-impact.md)
used generic v3. In the fixed-wing case, calibration headings covered roughly
two degrees and the inspection turn reached 132.51 degrees at familiar forward
speed. World-coordinate features changed substantially. The report does not
isolate orientation representation from the simultaneously changed rates and
other conditions, and it is not a measured current-v4 deficit.

## First iteration: establish the current Dart gap

Freeze a matched diagnostic using the existing ordinary recordings, explicit
train/development/evaluation roles, retained structured comparator and current
v4. Measure physical prediction errors, finite command responses and validity
at supported horizons before drawing an architectural conclusion. Retain the
same recordings and targets; do not collect paired simulator branches first.

The original terminal MPC has a default 1.2 s arrival plan with replanning every
30 ms. It consumes a 13-coordinate physical state plus actuator state and
uses differentiable recursive transitions. In that original comparator,
`control.py:156` obtains actual applied thrust from `plant.applied_command(state)`
at each replan. The historical result therefore includes simulator-provided
actuator information; it must not be relabeled as a history-only consumer trial.
Current v4 instead requires 500 ms
of observed history, returns 15 observation coordinates without position, and
limits each public forecast to 250 ms. Its existing Dart consumer exercises
short forecasts and selected JVPs, not the complete contact controller.

Resolve the consumer seam explicitly: causal measured history, physical command
units and motor order, differentiable position/attitude reconstruction, and
the exact candidate-command trajectory sent into the predictor. Freeze any
recursive use of predicted history for longer planning as an unqualified
extrapolation under test. Never pass simulator actuator truth into the learner,
silently shorten the controller horizon, pad fake observed history or interpret
short-horizon JVP agreement as physical command-gradient fidelity. If the public
interface cannot faithfully support the task, record that limitation as an
outcome and separate an explicitly declared research rollout from public support.

Keep the task, controller objective, command bounds, solve budget and truth
scorer fixed for matched model comparisons. Declare the information available
to each model, including actuator initialization. A fair history-only comparison
must give the structured arm a causal actuator estimate too; if that changes
its historical result, report it separately rather than carrying over the
1.47 mm score. Charge startup/priming consistently to both arms and record any
unavoidable interface difference. Use the model's
predictions in the actual objective; an output-only forecast demo is insufficient.
Preserve crashes, missed targets, optimizer failures, physical errors and timing.
Specify all of these and success criteria before fitting or controller trials.

## Leading architecture hypothesis: shared mechanics, learned response

Use a common mechanical state transition for all tested vehicles. Integrate
position from velocity and attitude from angular rate using valid rotations.
Express the learned response in body coordinates, transform it into world
coordinates, and include gravity explicitly. Learn effective non-gravitational
linear acceleration, angular acceleration and hidden response from observed
motion and commands. Effective accelerations avoid claiming that mass, inertia
and force scales can always be identified separately from the same recordings.

Command dimension comes from the recording contract. The learner receives no
vehicle-family selector, prescribed thrust axis, fixed rotor count, mixer,
aerodynamic coefficient table or simulator hidden state. Actuator lag and other
history effects remain learned. Frames, orientation and physical signal meanings
are required semantics, not vehicle tuning options. Body coordinates must not
erase relevant gravity direction, measured wind or environmental context.

This is a coherent candidate mechanism, not an established explanation of all
remaining errors. Derive its actual parameterization and freeze one comparison
after the Dart diagnostic. Use both Crazyflow and Cascade to detect a disguised
quadrotor assumption. More complicated contacts, articulation or flexible-body
effects need their own coverage evidence; shared rigid-body mechanics alone
does not establish support for every possible configuration.

Related work supports the direction: Duong et al.'s
[port-Hamiltonian neural dynamics](https://arxiv.org/abs/2401.09520) combines
learned dynamics with pose geometry and explicitly modeled dissipation across
robot examples. That supports using physical structure with learning; it does
not establish Glassbox performance or require adopting that full formulation.
For flight, drag, actuation and other nonconservative effects must remain
representable; imposing a purely energy-conserving model would be inappropriate.

## Decisions that prevent another open-ended tuning loop

If current v4 already meets the fixed Dart task, complete its consumer integration
and extend coverage before replacing architecture. If the interface or planning
horizon is the blocker, address that named gap before blaming model capacity.
If physically inconsistent/generalization errors remain, test the shared
mechanical formulation with the same ordinary data and fixed controller.
Accept a successor for measured forecast/response and task benefit under frozen
limits, with separate costs and losses; keep v4 if it does not earn adoption.
Paired response supervision remains a possible later diagnostic for a specific
unresolved failure. No extra data, family branch or parameter sweep is the
automatic next step.
