# Current state

Updated 2026-09-23. Read [the charter](charter.md) first. The **causal actuator
learner** is the single maintained implementation. One no-pretraining recipe
fits a shared nonlinear applied-command state, body force, rigid-body torque,
inertia and actuator momentum from each configuration's own observations. It
requires no vehicle family, mixer or actuator layout. The public
`fit`/`predict`/`update` contract, immutable fingerprinted revisions,
differentiable command predictions and background `OnlineFit` publication are
implemented. Superseded dynamics modules and experiments have been removed from
the working tree; Git retains their history. See the [learner contract](learner.md).

The corrected [matched-prefix and publication result](causal-publication.md)
uses the frozen eight-case, 263-origin benchmark. Every fit now starts at the
same recording row as the incumbent. Fully refitted 250 ms velocity **and**
body-rate RMSE improve on both fixed-wing recordings and all three active quad
arm recordings. Six arm-125 counterfactual command-response probes average
**0.073** relative error versus **0.331** for the incumbent. A generic weak
force-offset prior removes a severe early fixed-wing cancellation fit without
changing the equation or adding a platform branch.

In a paced post-initialization replay, the revisions actually published by a
background worker still beat the incumbent on both 250 ms components for the
two fixed-wing and the three longer active quad streams. The short 1.35-arm
stream loses velocity (0.802 versus 0.596 m/s) while cutting its rate error
from 6.237 to 0.828 rad/s. The two gentle paired flights retain small absolute
angular losses. Published models average roughly 0.4 s behind observations on
fixed-wing and 1.1–1.3 s on the longer quad streams. Initial fits take
0.12–0.38 s and are **not** overlapped with prefix collection in that replay;
first-origin cold-start availability is unqualified. A repeat of three
timing-sensitive cases preserves the broad decision, with score variation.
An exploratory CPU high-spin fixture check puts repeated 250 ms predictions at
about 0.4–0.8 ms after compilation; a new episode-history length within an
already compiled size bucket takes about 4–5 ms. First compilation of a new
bucket took roughly 0.1–0.26 s. These are predictor measurements, not a
controller deadline or hardware-independent guarantee.

This is direct prediction and command-response evidence on known configurations,
not a live Throw or Dart success claim. The model has no calibrated error
envelope. Measurement-noise tolerance, genuinely unseen configurations,
fixed-wing counterfactual response, predictor timing in a controller and the
complete cold-start timeline remain open. The previous
[full-state research result](causal-fullstate.md) used extra earlier causal rows
on some quad cases and is superseded for equal-data comparisons.

## Next iteration

The named gap is **held-out configuration accuracy of the single public recipe**.
Fit it from a fresh start on recordings outside the current eight-case cohort,
including the difficult Crazyflow arm conditions and Cascade or Dart flight
conditions when available. Measure physical full-state forecasts and matched
command-response interventions without vehicle metadata or tuning. Report data
and fitting provenance, noise and regime limits, then use those results to guide
one structural model change. Background cadence remains measured, but an
arbitrary per-observation CPU deadline is not an architecture gate.
