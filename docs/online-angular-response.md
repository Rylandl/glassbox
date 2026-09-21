# Causal angular response under online v6

The frozen diagnostic reproduced **all450 fixed-wing forecasts and updates exactly** and recovered23 actual pre-assimilation models. The quadratic head still dominates several angular spikes; its small final-model penalty did not describe these earlier states. The learner stayed byte-identical to adopted v6 throughout this diagnosis.

| Known origin | Native rate error (rad/s) | 64x refinement | Native quadratic pitch increment (rad/s) |
| --- | ---: | ---: | ---: |
| FW80 row31 | 5.135907 | 5.062818 | -4.263913 |
| FW81 row64 | 18.226968 | 13.199755 | -26.375057 |
| FW81 row150 | 2.120894 | 1.263334 | -43.420300 |
| FW81 row157 | 2.069978 | 0.911749 | -41.305300 |

Refinement leaves substantial model error. At row150 current-linear and delayed-linear pitch increments are +15.112 and +27.265rad/s, largely canceling the quadratic head. Signed trajectory accounting does not identify causal head error: deleting a head changes the integrated path. Local head derivatives freeze history, filters and hidden state; they do not establish recurrent stability.

Posthoc inspection found that v6 scales its prior to typical RMS excitation, while supported motion reaches much farther and commands can remain inside the observed range yet substantially exceed RMS. At FW81 row64 the largest quadratic angular derivative is pitch acceleration versus yaw rate, not pitch self-feedback. Its quadratic pitch increment splits -9.646rad/s from motion-involving terms and -16.729rad/s from other terms, motivating command-domain coverage as well as motion-domain coverage.

The next frozen test is [online v7](harness/online-fit-v7.json): use the existing supported-motion bounds and bounded-cache observed command maxima for the same quadratic prior. It keeps scalar strength, initialization, inference, data loss and optimization budget unchanged. This is a testable prior-domain correction, not a proven remedy or a physical command limit.

The protocol selected angular/velocity maxima and nearby ordinary predictions from known v6 outcomes. All450 transitions were replayed; all23 captures were verified. Factor1/2/4/16/64 stages, component angular increments and separate velocity/rate/attitude derivative blocks are retained. Official verification performs read-only model calls and zero optimizer steps. An independent NumPy audit checks370 payloads,12006 stage states and178273 assertions with no Glassbox/JAX calls. All declared absolute-plus-relative tolerances pass. The219-test suite and Ruff pass.

[Evidence index](online-angular-response.json) records frozen protocol/source commits, sealed artifact authority and supplemental audit hashes. This known-tape diagnosis establishes neither blind generalization nor current-learner closed-loop recovery.
