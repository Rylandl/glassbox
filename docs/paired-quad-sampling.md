# Paired quad observation schedules

The same Crazyflow quad motion and issued commands were observed at 10 and
50 ms. Commands changed every 50 ms and were held across the five intervening
10 ms plant steps. The 50 ms tape selects exact states from the 10 ms tape, so
the plant trajectory and issued commands are identical. Three unchanged,
fresh-start learners were fit independently on each tape: the maintained full
learner, curvature-regularized direct readout, and first-order sensitivity
direct readout. No fitted state crossed schedules. All scored predictions were
made before the next observation was assimilated.

The first collection protocol used the existing Throw pass-6 controller. The
vehicle hit the floor at 1.95 s and yielded only two common forecast origins.
This failed the diagnostic's useful-length requirement and is retained as an
attempt, not silently replaced. A revised fixed 10 s pilot used a deterministic
hover-and-sinusoid controller with Crazyflow model parameters solely to issue
commands. The learner never received those parameters. Collection succeeded
without clipping or floor contact. Maximum speed was 1.54 m/s, maximum body
rate 0.171 rad/s, and each motor command spanned about 0.035 of its full range.
This is a mild, narrow near-hover regime, not a comparable aerodynamic or
tumbling challenge to the Cascade recordings.

The evaluation protocol was committed before any of its six fits. Each learner
received the same 0.5–1.25 s elapsed prefix within its schedule. The 10 ms
schedule then assimilated 875 causal updates; the 50 ms schedule assimilated
175. Common 50–250 ms forecasts were scored at 35 matching physical origins,
from 1.25 to 9.75 s. Native one-step scores were kept separate because they
represent different elapsed horizons.

| Schedule and learner | 250 ms velocity RMSE (m/s) | Body-rate RMSE (rad/s) | Orientation RMSE (rad) | Warm complete update (ms) |
| --- | ---: | ---: | ---: | ---: |
| 10 ms full | 1.44943 | 0.13752 | 0.01223 | 33.47 |
| 10 ms curvature | 0.003119 | 0.007854 | 0.000700 | 1.011 |
| 10 ms sensitivity | 0.001637 | 0.005101 | 0.000528 | 1.324 |
| 50 ms full | 0.82055 | 0.13702 | 0.01166 | 8.440 |
| 50 ms curvature | 0.003431 | 0.004460 | 0.000498 | 0.574 |
| 50 ms sensitivity | 0.003390 | 0.004420 | 0.000482 | 0.654 |

The sensitivity readout has no fixed-wing-like body-rate explosion on this
50 ms quad recording: at every common horizon its body-rate RMSE is below
0.0045 rad/s; the largest individual 250 ms rate error is 0.0258 rad/s. This
rules out a 50 ms observation interval *by itself* as sufficient to cause the
previous fixed-wing divergence under the tested regime. It does not separate
sample interval from fivefold fewer updates, different history parameterization
or excitation; they change together. Nor does it establish broad quad
robustness from one gentle trajectory.

The maintained full learner has larger recursive error even on this simple
flight despite modest one-step error. Its 10 ms native one-step velocity RMSE is
0.00408 m/s, but 250 ms velocity RMSE reaches 1.45 m/s. The fast readout's
advantage here is absolute physical accuracy as well as runtime. Total complete
update CPU per elapsed recorded second is 3.70 s for the full learner versus
0.161 s for the sensitivity readout at 10 ms; at 50 ms it is 0.411 versus
0.0409 s. Initialization and first-use compilation were recorded separately
and are not included in warm medians. These times were measured on one local
CPU/runtime and are not portable real-time guarantees.

The authenticated evaluation pack contains every prediction, physical truth,
origin, update duration, model fingerprint, initial/final model, readout
features/targets/penalties and terminal sufficient statistics. Its independent
verifier recomputes physical metrics and timings from saved arrays, checks
recording rows/truth and final model fingerprints, and does no fitting. The
collection and evaluation protocols, source bindings, attempt and manifest
hashes are in [the index](paired-quad-sampling.json). No production learner or
public interface changed.
