# Dual-control NMPC release ensemble

`glassbox crazyflow throw-study --ensemble` flies every dual-control arm and
both cascade references on the same seeded distribution of perturbed throw
releases. It exists because seven deterministic releases cannot tell two
controller designs apart when one clipped coefficient at interval three flips a
case, and because the design page's recovery claims need an interval rather
than a count.

This page records the fifth-pass ensemble. The design, the diagnosis it answers
and the reading of its failure are in
[Dual-control NMPC](../concepts/dual-control-nmpc.md); this page is the
measurement.

## Protocol

Each of the seven throw-study cases is drawn sixteen times. Each world velocity
component is scaled by an independent `U[0.8, 1.2]`, each body rate component
likewise, and the release attitude is rotated by `U[0, 0.1]` radians about a
uniformly random horizontal body axis. The release height, the hidden airframe,
the loop rate and every controller setting are unchanged, so the altitude budget
the arms are spending stays comparable across draws.

Each draw is seeded from the triple `(ensemble seed, case index, replicate
index)`, so a single row reproduces on its own, adding a case or a replicate
never moves the others, and every arm flies exactly the same draws. Every
comparison below is therefore paired.

Recovery is the design page's own criterion: the hover envelope reached and the
floor never touched. The envelope alone is satisfied by a vehicle resting on the
ground, which is what makes the altitude test load-bearing. Intervals are Wilson
95 percent score intervals, because at sixteen releases the normal interval for
zero recoveries has exactly zero width.

Five arms, 112 releases each, 560 trials. The report is
`artifacts/crazyflow_throw_study/report-pass5-ensemble.json`.

A release the simulator cannot integrate to the end is recorded as a diverged
trial and counted as not recovered, rather than ending the run. It contributes
to no other statistic, and the arm's `simulator_diverged_count` and
`all_values_finite_and_bounded` are what make it visible. Only `pass5` produced
any.

## Reproducing the four earlier arms

The four arms the fourth pass measured were re-run unchanged beside the fifth.
All 448 of their trial records are byte-identical to
`report-pass4-ensemble.json`, and their per-case and pooled summaries are
identical except for the two keys this run adds
(`simulator_diverged_count`, and an `available_count` on the early-collective
block). The fifth pass therefore adds a column to the fourth pass's own table
rather than replacing it.

## Pooled result

| arm | recovered | rate | Wilson 95 | hover envelope | on the floor | diverged |
| --- | --- | --- | --- | --- | --- | --- |
| certified | 61/112 | 0.545 | 0.452-0.634 | 82 | 38 | 0 |
| working | 25/112 | 0.223 | 0.156-0.309 | 39 | 37 | 0 |
| pass2b | 7/112 | 0.062 | 0.031-0.123 | 37 | 99 | 0 |
| pass4 | 2/112 | 0.018 | 0.005-0.063 | 10 | 107 | 0 |
| pass5 | 0/112 | 0.000 | 0.000-0.033 | 12 | 109 | 3 |

`pass5` recovers nothing. Its Wilson upper bound is `0.033`, which excludes
`pass2b`'s point estimate of `0.062`, so the fifth pass is worse than the second
by this protocol and not merely indistinguishable from it. The two cascade arms
differ from each other by the same protocol, `0.545` against `0.223`, so the
ensemble has the resolution to detect a design change that matters.

| arm | early collective | rank four s med/worst | terminal tilt med/worst | settled step med/worst |
| --- | --- | --- | --- | --- |
| certified | 0.717 | 0.23/0.95 | 0.0009/1.1975 | 0.0010/0.1000 |
| working | 0.717 | 0.23/0.95 | 0.0276/1.2915 | 0.0723/0.1000 |
| pass2b | 0.314 | 0.32/4.65 | 0.0202/2.6575 | 0.1802/0.5474 |
| pass4 | 0.410 | 0.48/5.24 | 0.0804/2.7922 | 0.2258/0.9869 |
| pass5 | 0.044 | 0.96/7.22 | 1.2232/3.1351 | 0.0006/0.1000 |

Three of these columns are the whole story.

The first-`0.3 s` mean collective is `0.044` of the command range. `pass5`
commands almost no thrust while the vehicle is falling. Every other arm in the
study, including the two the fourth pass showed were spending too little, is
between seven and sixteen times higher.

The median time to command information rank four is `0.96 s`, three times
`pass2b`'s and four times the cascade's, and 13 of the 112 releases never reach
rank four at all. `pass5` is the first arm in this study that fails to identify
the command map on some releases.

The median terminal tilt is `1.22 rad`. That is a vehicle lying on its side.
The worst of the other four arms is `0.0804 rad` on the median release, and the
best is `0.0009`.

The fourth column is the trap. `pass5` has the quietest settled command of any
arm measured on this diagnostic — pooled median step `0.0006` of the command
range, below the frozen snapshot's `0.0010` — on 109 flights that are motionless
on the floor. It is the clearest demonstration this study has produced that the
chatter criterion is only meaningful after the recovery criterion is met.

## Per case and arm

| case                            | arm                      | recovered | rate | wilson 95 | early collective | terminal speed med/worst | terminal rate med/worst | terminal tilt med/worst | min alt med/worst | rank four s med/worst | set cmd step med/worst |
| ---                             | ---                      | ---       | ---  | ---       | ---              | ---                      | ---                     | ---                     | ---               | ---                   | ---                    |
| canonical                       | certified                | 7/16      | 0.44 | 0.23-0.67 | 0.786            | 0.011/0.601              | 0.013/1.187             | 0.0011/1.1975           | 0.239/-0.001      | 0.29/0.95             | 0.0010/0.1000          |
| canonical                       | working                  | 2/16      | 0.12 | 0.03-0.36 | 0.786            | 0.096/0.601              | 0.255/1.187             | 0.0390/1.1975           | 0.368/-0.001      | 0.29/0.95             | 0.0802/0.1000          |
| canonical                       | dual_control_nmpc_pass2b | 1/16      | 0.06 | 0.01-0.28 | 0.369            | 0.008/2.356              | 0.104/0.859             | 0.0272/0.1153           | -0.001/-0.001     | 0.28/0.41             | 0.1697/0.3375          |
| canonical                       | dual_control_nmpc_pass4  | 0/16      | 0.00 | 0.00-0.19 | 0.386            | 0.002/0.908              | 0.545/31.584            | 0.1434/2.1388           | -0.001/-0.001     | 0.55/1.38             | 0.2161/0.9869          |
| canonical                       | dual_control_nmpc_pass5  | 0/16      | 0.00 | 0.00-0.19 | 0.028            | 0.003/0.419              | 0.551/44.204            | 0.7408/3.1261           | -0.001/-0.001     | 0.95/5.28             | 0.0010/0.1000          |
| shorter_arms_high_release       | certified                | 11/16     | 0.69 | 0.44-0.86 | 0.754            | 0.012/0.150              | 0.013/0.991             | 0.0010/0.6425           | 2.500/-0.001      | 0.23/0.76             | 0.0010/0.1000          |
| shorter_arms_high_release       | working                  | 5/16      | 0.31 | 0.14-0.56 | 0.754            | 0.020/0.667              | 0.062/0.991             | 0.0147/1.2915           | 2.500/-0.001      | 0.23/0.76             | 0.0742/0.1000          |
| shorter_arms_high_release       | dual_control_nmpc_pass2b | 1/16      | 0.06 | 0.01-0.28 | 0.328            | 0.011/10.407             | 0.190/1.995             | 0.0215/2.2116           | -0.001/-0.001     | 0.34/0.57             | 0.2082/0.4193          |
| shorter_arms_high_release       | dual_control_nmpc_pass4  | 0/16      | 0.00 | 0.00-0.19 | 0.272            | 0.000/2.467              | 0.433/25.929            | 0.1517/2.5246           | -0.001/-0.001     | 0.79/5.24             | 0.1916/0.8728          |
| shorter_arms_high_release       | dual_control_nmpc_pass5  | 0/16      | 0.00 | 0.00-0.19 | 0.014            | 0.018/4.880              | 2.308/63.435            | 0.9583/3.0638           | -0.001/-0.001     | 1.07/7.22             | 0.0025/0.1000          |
| long_arms_cross_axis_tumble     | certified                | 9/16      | 0.56 | 0.33-0.77 | 0.726            | 0.017/0.215              | 0.008/1.230             | 0.0005/0.7240           | 0.466/-0.001      | 0.23/0.39             | 0.0010/0.1000          |
| long_arms_cross_axis_tumble     | working                  | 2/16      | 0.12 | 0.03-0.36 | 0.726            | 0.098/0.377              | 0.202/1.230             | 0.0228/0.7240           | 0.753/-0.001      | 0.23/0.39             | 0.0757/0.1000          |
| long_arms_cross_axis_tumble     | dual_control_nmpc_pass2b | 2/16      | 0.12 | 0.03-0.36 | 0.421            | 0.008/0.776              | 0.112/0.451             | 0.0149/0.0609           | -0.001/-0.001     | 0.29/0.45             | 0.0808/0.4421          |
| long_arms_cross_axis_tumble     | dual_control_nmpc_pass4  | 2/16      | 0.12 | 0.03-0.36 | 0.531            | 0.005/2.405              | 0.323/1.472             | 0.0253/2.6219           | -0.001/-0.001     | 0.51/1.24             | 0.2283/0.4142          |
| long_arms_cross_axis_tumble     | dual_control_nmpc_pass5  | 0/16      | 0.00 | 0.00-0.19 | 0.057            | 0.020/3.146              | 1.112/11.321            | 2.0755/3.1310           | -0.001/-0.001     | 1.03/4.18             | 0.0003/0.1000          |
| milder_low_energy_release       | certified                | 16/16     | 1.00 | 0.81-1.00 | 0.732            | 0.011/0.013              | 0.012/0.012             | 0.0009/0.0010           | 2.284/0.937       | 0.23/0.31             | 0.0010/0.0010          |
| milder_low_energy_release       | working                  | 9/16      | 0.56 | 0.33-0.77 | 0.732            | 0.032/0.301              | 0.029/0.245             | 0.0055/0.0809           | 2.272/0.931       | 0.23/0.31             | 0.0400/0.0860          |
| milder_low_energy_release       | dual_control_nmpc_pass2b | 2/16      | 0.12 | 0.03-0.36 | 0.381            | 0.026/0.473              | 0.048/5.960             | 0.0125/0.8044           | -0.001/-0.001     | 0.28/0.63             | 0.0549/0.5474          |
| milder_low_energy_release       | dual_control_nmpc_pass4  | 0/16      | 0.00 | 0.00-0.19 | 0.419            | 0.001/0.368              | 0.445/0.926             | 0.1218/1.6939           | -0.001/-0.001     | 0.49/0.98             | 0.2171/0.5275          |
| milder_low_energy_release       | dual_control_nmpc_pass5  | 0/16      | 0.00 | 0.00-0.19 | 0.110            | 0.013/3.686              | 0.684/30.664            | 2.1095/3.1292           | -0.001/-0.001     | 0.67/1.97             | 0.0007/0.1000          |
| reversed_tumble                 | certified                | 6/16      | 0.38 | 0.18-0.61 | 0.732            | 0.021/0.209              | 0.009/0.694             | 0.0010/1.0283           | -0.001/-0.001     | 0.25/0.38             | 0.0010/0.1000          |
| reversed_tumble                 | working                  | 2/16      | 0.12 | 0.03-0.36 | 0.732            | 0.010/0.360              | 0.034/0.947             | 0.0037/1.0283           | -0.001/-0.001     | 0.25/0.38             | 0.0733/0.1000          |
| reversed_tumble                 | dual_control_nmpc_pass2b | 0/16      | 0.00 | 0.00-0.19 | 0.278            | 0.006/10.601             | 0.111/2.063             | 0.0225/2.6575           | -0.001/-0.001     | 0.34/1.33             | 0.1921/0.4512          |
| reversed_tumble                 | dual_control_nmpc_pass4  | 0/16      | 0.00 | 0.00-0.19 | 0.410            | 0.009/5.743              | 0.437/5.402             | 0.0524/2.1691           | -0.001/-0.001     | 0.58/1.36             | 0.2296/0.5776          |
| reversed_tumble                 | dual_control_nmpc_pass5  | 0/16      | 0.00 | 0.00-0.19 | 0.020            | 0.010/1.302              | 0.473/38.947            | 1.3657/3.1274           | -0.001/-0.001     | 1.02/6.13             | 0.0007/0.1000          |
| canonical_state_noise           | certified                | 2/16      | 0.12 | 0.03-0.36 | 0.511            | 0.278/6.520              | 0.021/0.812             | 0.0023/0.8895           | 1.033/-0.001      | 0.23/0.23             | 0.0198/0.1000          |
| canonical_state_noise           | working                  | 1/16      | 0.06 | 0.01-0.28 | 0.511            | 0.357/0.664              | 0.391/1.185             | 0.0396/0.8895           | 1.032/-0.001      | 0.23/0.23             | 0.0652/0.1000          |
| canonical_state_noise           | dual_control_nmpc_pass2b | 0/16      | 0.00 | 0.00-0.19 | 0.069            | 0.022/2.613              | 0.325/18.350            | 0.0900/1.2475           | -0.001/-0.001     | 0.58/4.65             | 0.2598/0.4710          |
| canonical_state_noise           | dual_control_nmpc_pass4  | 0/16      | 0.00 | 0.00-0.19 | 0.450            | 0.004/2.208              | 0.706/1.406             | 0.1919/2.7922           | -0.001/-0.001     | 0.36/0.58             | 0.3163/0.8185          |
| canonical_state_noise           | dual_control_nmpc_pass5  | 0/16      | 0.00 | 0.00-0.19 | 0.043            | 0.000/0.049              | 0.274/1.564             | 0.2763/3.1290           | -0.001/-0.001     | 0.70/3.01             | 0.0005/0.1000          |
| canonical_mid_flight_arm_change | certified                | 10/16     | 0.62 | 0.39-0.82 | 0.779            | 0.010/0.057              | 0.011/2.376             | 0.0005/1.0038           | 1.200/-0.001      | 0.23/0.74             | 0.0010/0.1000          |
| canonical_mid_flight_arm_change | working                  | 4/16      | 0.25 | 0.10-0.49 | 0.779            | 0.039/0.342              | 0.160/2.376             | 0.0248/1.0038           | 1.200/-0.001      | 0.23/0.74             | 0.0757/0.1000          |
| canonical_mid_flight_arm_change | dual_control_nmpc_pass2b | 1/16      | 0.06 | 0.01-0.28 | 0.356            | 0.006/0.304              | 0.061/0.513             | 0.0165/0.0819           | -0.001/-0.001     | 0.32/0.51             | 0.1609/0.2819          |
| canonical_mid_flight_arm_change | dual_control_nmpc_pass4  | 0/16      | 0.00 | 0.00-0.19 | 0.406            | 0.001/0.207              | 0.570/1.407             | 0.0564/2.1619           | -0.001/-0.001     | 0.42/1.07             | 0.1991/0.6551          |
| canonical_mid_flight_arm_change | dual_control_nmpc_pass5  | 0/16      | 0.00 | 0.00-0.19 | 0.028            | 0.000/0.025              | 0.122/88.492            | 2.2985/3.1351           | -0.001/-0.001     | 1.02/3.86             | 0.0002/0.1000          |
| pooled                          | certified                | 61/112    | 0.54 | 0.45-0.63 | 0.717            | 0.011/6.520              | 0.012/2.376             | 0.0009/1.1975           | 1.200/-0.001      | 0.23/0.95             | 0.0010/0.1000          |
| pooled                          | working                  | 25/112    | 0.22 | 0.16-0.31 | 0.717            | 0.078/0.667              | 0.180/2.376             | 0.0276/1.2915           | 1.200/-0.001      | 0.23/0.95             | 0.0723/0.1000          |
| pooled                          | dual_control_nmpc_pass2b | 7/112     | 0.06 | 0.03-0.12 | 0.314            | 0.010/10.601             | 0.115/18.350            | 0.0202/2.6575           | -0.001/-0.001     | 0.32/4.65             | 0.1802/0.5474          |
| pooled                          | dual_control_nmpc_pass4  | 2/112     | 0.02 | 0.00-0.06 | 0.410            | 0.002/5.743              | 0.466/31.584            | 0.0804/2.7922           | -0.001/-0.001     | 0.48/5.24             | 0.2258/0.9869          |
| pooled                          | dual_control_nmpc_pass5  | 0/112     | 0.00 | 0.00-0.03 | 0.044            | 0.014/4.880              | 0.488/88.492            | 1.2232/3.1351           | -0.001/-0.001     | 0.96/7.22             | 0.0006/0.1000          |

`pass5` is the worst arm on every case. There is no case, and no perturbation
of any case, on which it recovers.

## Within-arm time to command rank four

The fourth pass found that the quantity separating recovered from lost releases
*inside* an arm is how quickly the command evidence reaches rank four, and that
the first-`0.3 s` mean collective is a between-arm association rather than a
mechanism. The fifth pass cannot test that split, because it has no recovered
releases to compare against.

Median seconds after enable, recovered against lost:

| arm | recovered | lost | never |
| --- | --- | --- | --- |
| certified | 0.230 (n=61) | 0.230 (n=51) | 0 |
| working | 0.230 (n=25) | 0.230 (n=87) | 0 |
| pass2b | 0.280 (n=7) | 0.320 (n=105) | 1 |
| pass4 | 0.375 (n=2) | 0.485 (n=110) | 0 |
| pass5 | n/a (n=0) | 0.960 (n=112) | 13 |

What `pass5` does add is the other end of the same association. It is the
slowest arm to rank four by a factor of three, it is the only arm that fails to
reach rank four at all on some releases, and it is the only arm with a zero
recovery rate. That is consistent with the fourth pass's reading and adds no
independent support for it, because a controller that commands almost no thrust
identifies slowly and crashes for the same underlying reason.

## What the videos show

`artifacts/crazyflow_throw_study/videos/canonical-pass5.mp4` and
`reversed-tumble-pass5.mp4`, with their contact sheets, are the deterministic
releases of two cases.

On the canonical release the vehicle is thrown, reaches apex around one second,
and then falls in a straight vertical line to the floor, arriving before the
two-second mark. There is no arrest attempt in the trace: no lateral excursion,
no visible attitude correction, no motor transient. From three seconds to the
end of the ten-second window it lies still on the ground, roughly upright, and
does not move again. The frames at five and ten seconds are indistinguishable
from each other.

The contrast with the recorded `pass4` video of the same release is the useful
one. `pass4` also ends on the floor, but it is visibly fighting: the trace
after impact shows continuing motion, the airframe tumbles, and the flight
traces sweep across several tiles. `pass5` does not fight at all. Watching them
side by side, `pass4` looks like a controller with the wrong model and `pass5`
looks like a controller that has been switched off.

`reversed-tumble-pass5.mp4` is the same shape. The fall is again a straight
vertical line with no arrest attempt, the vehicle is on the ground by two
seconds, and the difference is only in how it comes to rest: it rocks onto its
side rather than settling upright, and the remaining eight seconds are spent
lying at an angle. Neither video contains a recovery attempt that failed;
neither contains a recovery attempt.

That is a faithful picture of the objective. The design page explains why: at
zero information the planned-trajectory spread charge scores holding the current
command strictly better than any excitation, and with a zero command map the
tracking term cannot distinguish plans at all, so the cheapest plan on the board
is to do nothing. The controller is not failing to solve its problem. It is
solving it, and the solution is to keep the motors off.

## Sixth pass, arm-only ensembles

The sixth pass was measured one arm at a time on the same seeded releases, so
its rows pair with this page's table by construction. The reports are
`artifacts/crazyflow_throw_study/report-pass6-ensemble-arm-only.json`
(`pass6` as committed), `report-pass6-pre-switch-ensemble-arm-only.json`
(the same design before the switches that keep the fifth pass bit-identical),
`report-pass6-motor-basis-ensemble-arm-only.json` (motor-basis probes), and
`report-pass6-round2-block-shift-ensemble-arm-only.json` and
`report-pass6-round2-step-shift-ensemble-arm-only.json` (round two: the
knowledge term over the posterior's hover distribution, with the block and the
real-time warm start), and `report-pass6-round2-ensemble-arm-only.json` (the
committed round two, with the composite seeds).

| arm | recovered | rate | Wilson 95 | on the floor | airborne, outside envelope |
| --- | --- | --- | --- | --- | --- |
| pass6, motor-basis probes | 14/112 | 0.125 | 0.076-0.199 | 48 | 50 |
| pass6, before the switches | 11/112 | 0.098 | 0.056-0.167 | 50 | 51 |
| pass6, round one | 10/112 | 0.089 | 0.049-0.157 | 52 | 48 |
| pass6, round two, block warm start | 47/112 | 0.420 | 0.332-0.512 | 51 | 9 |
| pass6, round two, real-time warm start | 47/112 | 0.420 | 0.332-0.512 | 61 | 1 |
| pass6, round two, with composite seeds (committed) | 54/112 | 0.482 | 0.392-0.574 | 50 | 5 |

The round-one intervals exclude `pass5`'s point estimate and include
`pass2b`'s. The committed round two's interval, 0.392-0.574, excludes every
earlier learned arm and the working cascade (0.223) and contains the certified
cascade's point estimate (0.545). The design page records the configurations
that were measured on the single-release gate and dropped, and why.

## Limitations

- The perturbation is a declared distribution over the release state only; the
  hidden airframe, the loop rate and every controller setting are unchanged.
- Recovery is a binary read of one envelope on one ten-second window, so a
  marginal arrest and a comfortable one score the same.
- The state-noise case draws one noise realisation per release, not a
  distribution over realisations.
- A release the simulator could not integrate to the end is counted as not
  recovered and contributes nothing to any other statistic. Three `pass5`
  releases are in that state.
- The fifth pass changed the plan parameterization, the spread model, the seed
  family, the horizon and the rate limit together. This ensemble measures their
  joint effect and cannot attribute it.

## Reproduce

```bash
uv sync --dev --extra crazyflow --extra crazyflow-animation
uv run glassbox crazyflow throw-study \
  artifacts/crazyflow_throw_study/report-pass5-ensemble.json \
  --ensemble --workers <physical cores minus one>
uv run glassbox crazyflow throw-study \
  artifacts/crazyflow_throw_study/report-pass5.json \
  --control-model certified --control-model working \
  --control-model dual_control_nmpc_pass2b \
  --control-model dual_control_nmpc_pass4 \
  --control-model dual_control_nmpc_pass5
uv run glassbox crazyflow throw-animation \
  artifacts/crazyflow_throw_study/videos/canonical-pass5.mp4 \
  --gif artifacts/crazyflow_throw_study/videos/canonical-pass5.gif \
  --contact-sheet \
  artifacts/crazyflow_throw_study/videos/canonical-pass5-contact-sheet.png \
  --control-model dual_control_nmpc_pass5 --case canonical
```

The report is identical at any worker count: every trial is decided by its own
seeded release and its arm, and the results are reassembled in job order rather
than completion order.
