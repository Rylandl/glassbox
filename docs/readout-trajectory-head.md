# Causal trajectory correction of the fast readout

A single low-rank correction to the fast readout's acceleration head made the
first fixed-wing forecast far more accurate, but it also amplified an already
unstable hard-quad rollout. This is a useful architectural screen, not a
production replacement or a Throw qualification.

The starting model is the previously evaluated physical SO(3) direct readout.
Each fit begins fresh from the same causal recording prefix; its feature and
history parameters are frozen within that episode. After the prefix and each
new observation, the candidate uses the most recent **completed** 250 ms
physical trajectory. It differentiates endpoint velocity, body rate and
orientation residuals with respect to the entire acceleration readout head,
solves one Gram-scaled 15-dimensional dual ridge system, and accepts the first
backtracked correction that improves that completed endpoint. The direct head
is re-solved after every observation; the trajectory correction is recomputed
from it, never accumulated. No fleet pretraining, vehicle metadata, family
branch or future observation enters the correction. Source and tests were
committed before fitting.

The frozen screen used eight authenticated existing recordings/schedules, one
fresh fit per case and **126 post-prefix candidate updates**. It scored the
first and next previously frozen conditional origins at 50, 100, 150, 200 and
250 ms, plus every intervening native one-step forecast. The first prediction
follows the 0.75 s fixed-wing or 1.25 s quad prefix and includes the prefix
correction. The previous direct and full-learner controls were authenticated
existing forecasts at the same origins; their first-origin forecasts coincide
because they share the fresh initial model before this correction.

At the **first origin**, 250 ms errors were:

| Recording | Velocity: correction / direct (m/s) | Body rate: correction / direct (rad/s) |
| --- | ---: | ---: |
| Fixed wing 80 | **0.214 / 2.450** | **0.342 / 8.956** |
| Fixed wing 81 | **1.902 / 8.163** | **2.138 / 22.063** |
| Quad arm 115 | 2.066 / 2.435 | 7.138 / 5.787 |
| Quad arm 125 | 1.681 / 1.497 | 7.478 / 7.478 |
| Quad arm 135, truncated | **11.032 / 7.873** | **163.717 / 102.736** |
| Quad change | 1.681 / 1.497 | 7.478 / 7.478 |
| Paired quad 10 ms | 0.009 / 0.007 | 0.007 / 0.030 |
| Paired quad 50 ms | 0.011 / 0.019 | 0.006 / 0.026 |

The fixed-wing improvement is unusually large for a change to only the
readout. Its initial body-rate error falls by 96%/90%. In the truncated quad
case the opposite happens: body-rate error rises by 59% and velocity error by
40%. The difficult quad was already outside a useful 250 ms accuracy range;
this makes it worse, not merely a small isolated regression. At the second
origin, fixed-wing 250 ms rate is 0.798/2.079 rad/s with correction versus
0.790/2.082 for direct. The truncated quad is 18.081 versus 16.967 rad/s.
Other second-origin differences are mixed and often small. This short screen
does not establish sustained online benefit.

The saved horizon series narrows the failure. In the truncated quad's first
forecast, corrected/direct rate errors are 2.263/2.223 at 50 ms,
12.971/14.724 at 100 ms, 39.066/36.089 at 150 ms,
108.720/58.568 at 200 ms, and 163.717/102.736 at 250 ms. The completed
window used for its prefix correction improves, but the next trajectory
diverges. A separate no-fit diagnostic found that the correction also improves
recent completed 50 ms rate errors in this case. Consequently a gate checking
only teacher-forced or near-term completed data would not have caught the
unseen long-horizon failure. The dual correction is small in normalized
coefficient norm and the specified unit trust cap never binds; that norm does
not bound the resulting recursive forecast change.

The complete warm-update medians were **1.59–1.62 ms** on fixed wing,
**4.09–4.52 ms** on the 10 ms quad recordings and **2.31 ms** on the paired
50 ms quad. These include the direct solve, rollout differentiation, dual
solve, backtracking and model snapshot. First JAX compilation appears in
initialization or the first update and can cost seconds; the prefix cannot be
treated as a free calibration period. The earlier direct and full-learner
timings were separate-run references, not paired CPU measurements.

The saved-data verifier re-authenticates all input and control packs, checks
that every direct solve exactly matches the prior run, recomputes the physical
scores and timing summaries, and independently rebuilds all 15-dimensional
dual corrections, trust norms and accepted backtracking steps from saved
arrays without fitting. The Jacobian was checked against finite differences
for three- and four-command cases. The first planned attempt stopped after
one case because the evaluator reused a saved-array key with different
prefix/update lengths; the committed harness fix preceded the completed
eight-case run. Both attempts remain sealed in the [artifact index](readout-trajectory-head.json).

The next architectural test should fit the intermediate points of the
completed trajectory rather than its endpoint alone, then score the same
unseen horizons. The result should be judged on absolute physical error and
the hard-quad failure, not on the completed-window objective alone. Production
source remains unchanged; neither a new controller trial nor held-out vehicle
calibration has been performed.
