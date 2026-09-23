# Current state

Updated 2026-09-23. Read [the charter](charter.md) first. The supported
shared-physics learner with nonlinear history, accumulators and an
episode-fitted angular-rate memory remains the **single maintained public
implementation**. Its fit, differentiable predict, immutable update, save/load
and OnlineFit contracts are intact. It uses no pretraining or vehicle-family
input. [Public evidence](public-rate-memory.md).

The episode-fitted **causal actuator model is selected for integration**, not
yet public. It shares a nonlinear applied-command state between force and
torque and fits inertia and actuator momentum from the current episode. On the
complete frozen 263-origin benchmark it improves 250 ms velocity and body-rate
RMSE on both fixed-wing recordings and all three active quad-arm recordings.
On six arm-125 counterfactual probes, mean relative command-response error is
0.072 versus 0.331 for the public model. The difficult 0.85-arm high-spin rate
error falls from 27.360 to 0.410 rad/s at its 250 ms origin. The two gentle
paired flights lose body-rate accuracy by a few milliradians per second, and
the two earliest fixedwing-80 origins also regress. These losses remain visible
in the [full-state result](causal-fullstate.md).

The research fitter refits each causal prefix as a batch. It has no immutable
public revision, prediction derivative contract or background publication
cadence, so this is identification evidence rather than a live-controller
result. The current CPU's per-observation update time is **not** an adoption
gate. In a live episode, the model actually published at each moment must be
evaluated, with its data and elapsed fitting time counted. Unseen vehicle
classes, noise tolerance, fixed-wing counterfactual response and live Throw/Dart
outcomes remain open.

## Next iteration

The named gap is **public realization of the causal actuator model**. Replace
the existing dynamics implementation with one self-contained, differentiable
fit/predict/update revision and a causal background fitter that periodically
publishes it. Preserve the no-pretraining, no-metadata and variable-input
contracts. Qualify saved revisions, derivatives and actual publication timing
against the existing frozen recordings; do not add a second maintained model,
vehicle branch, consumer tuning option or arbitrary per-observation CPU limit.
