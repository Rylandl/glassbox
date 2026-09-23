# A physical-time actuator response trades early sensitivity for quad rollout

The previous [distributed-lag screen](readout-distributed-lag.md) fits
independent prompt and leaky effects per command, with the leaky time constant
fixed at eight recording intervals. This experiment instead fits one gain
direction per issued command. A shared prompt fraction and lag time in seconds
determine when that gain acts. At each update, a fixed grid of 0.02, 0.05,
0.10, 0.20 and 0.40 s lag times and prompt fractions 0, 0.25, 0.50, 0.75 and
1 is scored on the most recent completed 50 ms rate rollout; the winning pair
is refitted on the same latest 15–25 causal transitions. No vehicle type or
previous episode is used. The existing JAX force/rate rollout is unchanged
except that it receives the chosen physical lag. This remains an experimental
wrapper, not the public learner.

| Recording | 250 ms rate RMSE, previous → coupled (rad/s) | 250 ms velocity RMSE, previous → coupled (m/s) |
| --- | ---: | ---: |
| fixedwing-80 | 0.486 → **0.474** | **0.632** → 0.735 |
| fixedwing-81 | 0.599 → **0.482** | 1.198 → **0.794** |
| quad-arm-115 | **1.165** → 1.404 | 0.269 → **0.266** |
| quad-arm-125 | **0.624** → 0.928 | **0.130** → 0.153 |
| quad-arm-135 | 6.770 → **6.094** | **0.671** → 0.846 |
| quad-change | **0.624** → 0.928 | **0.130** → 0.153 |
| paired-quad-fine | 0.004 → 0.004 | **0.042** → 0.045 |
| paired-quad-coarse | 0.001 → 0.001 | 0.007 → 0.007 |

All 263 frozen origins and 4,187 online updates were evaluated. `quad-change`
largely repeats arm 125, and hard arm 135 has only three origins. The coupled
model improves the fixed-wing-81 and hard-quad rate forecasts, but loses the
main arm-125/115 quad gains. On the six arm-125 simulator command-response
probes, mean relative error is **0.379 → 0.478**. The first probe improves
**1.071 → 0.533**, while four later probes worsen: coupled errors are 0.533,
0.542, 0.290, 0.466, 0.366 and 0.673. In the two-origin early arm-125 smoke
screen, 250 ms rate RMSE worsens 2.718 → 4.594 rad/s. The coupled head
therefore does not meet the intended combination of early command accuracy
and stable quad recursion. Warm CPU update medians rise from roughly 0.6–1.5
ms to 3.5–7.3 ms, largely from repeated least-squares solves across the grid.
No controller trial or held-out configuration has qualified it.

The failure points to an identification tradeoff. Allowing separate prompt
and delayed effect vectors gives the quad rollout more freedom but can yield
large cancelling coefficients and a wrong first-step sensitivity. Coupling
those vectors improves the first measured sensitivity but removes useful
freedom for later response. Other exploratory **smoke-only** variants did not
resolve the tradeoff: selecting a physical lag with independent vectors
worsened early arm-125 and hard-quad rollouts; increasing the validation
horizon did not repair coupled arm-125; truncating the independent design's
small singular values lost its quad control signal; a full cross-axis rate
matrix produced divergent fixed-wing forecasts; and refitting the readout to
short trajectories raised cost and worsened early arm-125. None was run as a
full candidate or retained as source. The available 25-transition arm-125
prefix provides some excitation, but its eight prompt/delayed control columns
have a centered singular-value spread of about 340:1 at the first probe.
That diagnoses a sensitivity problem; it does not prove lag is unidentifiable
from every possible episode.

The coupled candidate is committed at `bc49fbf`. The final saved artifact is
`artifacts/readout-coupled-lag-v1/coupled-physical-lag-full`, manifest
`bc00e440a46b1aa299e032065710cbe166e0f79ef4ded585a9e5e326a8c0ce04`.
The frozen runner verified forecasts, controls, responses, scores and table
from saved data without refitting. Cleaning the experiment after its first
smoke run preserved every saved array exactly. The faster distributed-lag
experiment remains the accuracy reference; neither is adopted into public
`fit/predict/update` yet.
