# Causal learner on new Crazyflow flights

The single episode-fitted learner was scored on the old difficult 0.85/1.40-arm
recordings and on two newly collected flights. The new roster was committed in
`d891490` **before collection**. Its pinned Crazyflow 0.3.2 plant and Throw
pass6 policy generated the flight data independently of the candidate. The
learner saw only motion, issued commands and timing. Each fit started at row 50;
the first 250 ms forecast began at row 125 after 75 observed transitions, then
new fits were scored every 25 rows. Future issued commands were supplied for a
conditional forecast; the candidate did not fly either vehicle.

The 0.95-arm flight started at 2.4 m with an initial body rate of
`(2.4, -1.8, 1.2)` rad/s and reached 9.67 rad/s. The 1.55-arm flight used the
canonical 1.2 m release and reached 6.13 rad/s. Both completed 10 s without
floor contact. These are new arm lengths and flight conditions within the
Crazyflow family, not a new vehicle class. The earlier 0.85 and 1.40 flights
were already used to diagnose the former public model and are **historical
challenges**, not blind holdouts for this architecture.

| Recording | Origins | 250 ms velocity RMSE, original → prior (m/s) | 250 ms body-rate RMSE, original → prior (rad/s) | Hold-state velocity / rate (m/s / rad/s) |
| --- | ---: | ---: | ---: | ---: |
| Historical arm 0.85, high spin | 2 | 0.153 → 0.169 | 1.335 → 1.749 | 3.352 / 9.073 |
| Historical arm 1.40 | 35 | 0.0169 → 0.0154 | 0.0390 → 0.0406 | 0.971 / 1.165 |
| New arm 0.95, faster release spin | 35 | 0.0264 → 0.0298 | 0.130 → 0.159 | 0.627 / 3.329 |
| New arm 1.55, canonical release | 35 | 0.0784 → 0.0400 | 0.574 → 0.0909 | 1.154 / 1.553 |

“Original” is the public causal learner at `365b4af`. “Prior” adds the small
system-independent, data-decaying rise/fall prior in `9ddd3fb`. For context,
the *previous* public model scored 19.827 rad/s rate and 1.653 m/s velocity on
the historical 0.85 flight, and 0.455 rad/s and 0.436 m/s on 1.40. The causal
architecture itself therefore resolves the old high-spin divergence; the new
prior addresses a different early-data weakness.

At the first origin of arm 1.55, only 25 of the 75 fit transitions had nonzero
commands. The unrestricted fit inferred an almost stationary falling-command
rate (`0.47 + 0.11 (|target| + |applied|)` versus rising
`15.20 + 1.38 (|target| + |applied|)`, in reciprocal seconds), then moved to a
fast falling response after 25 more observations.
Its first 250 ms body-rate endpoint error was 3.382 rad/s. The prior reduces
that to 0.426 rad/s and the flight-wide RMSE by 6.3×. It penalizes the squared
log rise/fall rate ratios at zero and full command with weight `0.03 / n`,
where `n` is fitted transition count. It neither forces symmetric response nor
uses vehicle metadata. Exploratory fully symmetric and three-parameter shared
rate variants improved this first origin but substantially degraded the
high-spin 0.85 and 0.95 origins, so they were discarded.

The tradeoff is visible: the prior raises the first 0.95-arm rate endpoint error
from 0.550 to 0.769 rad/s and the first historical 0.85 endpoint from 1.842 to
2.437 rad/s. These regressions are retained as follow-up evidence, not hidden
behind the arm-1.55 aggregate gain. On the frozen, established 263-origin
eight-case suite, fixed-wing 250 ms rate RMSE changes only 0.299→0.299 and
0.350→0.350 rad/s; the largest full-suite rate increase is 0.110→0.115 rad/s
on the related arm-125 and configuration-change recordings. Both saved-model
verifiers replay the results without fitting, and the package's 51 tests pass.

In the frozen paced post-initialization replay, saved published revisions with
the prior still outperform the former public learner on both 250 ms velocity
and body rate for the two fixed-wing and three longer quad streams. The
arm-1.15 rate score is 0.287 rad/s, versus 0.222 in the earlier no-prior run;
repeating the prior gave 0.287, while a contemporary no-prior repeat gave
0.269. The revised fit had a roughly 1.36 s mean model age in those repeats,
versus 1.24 s for the contemporary no-prior run. The fully refitted arm-1.15
score scarcely changed (0.075→0.077), so the paced difference is mainly about
revision availability, and the historical 0.222 comparison includes timing
variation. The initial prefix fit is still synchronous in this replay, not a
qualified cold-start Throw timeline.

The source pack for the new flights is
`artifacts/fresh-causal-v1/source`, authenticated by manifest SHA-256
`43b2093288668697489e7d3e4394e3e1b65f1bb0081fba5436f6e0f8716be3ec`.
The historical source/evaluation manifest SHA-256 values are respectively
`5a60009ea431da8609186cb4cf3bd45419991724215ceeabf385b2cd9084ea80`
and `e516c24e78544cd3d758cb7b448aae432f5ebfe920b0c4eb3c9fd64d279cb713`.
Scores and all fitted revisions are in
`artifacts/fresh-causal-v1/evaluation`,
`artifacts/fresh-causal-prior-v1`, `artifacts/heldout-causal-v1` and
`artifacts/heldout-causal-prior-v1`. The frozen one-off scorers and collector
are in Git commits `a85a72e` and `d891490`; the selected prior is in `9ddd3fb`.
They were removed from the maintained working tree after verification. The new
recordings were examined when choosing the prior, so the **updated** model's
result is development-set evidence rather than untouched validation. One flight
per new condition,
perfect simulator observations, factual command sequences, refits at every
origin and no live controller trial leave noise tolerance, counterfactual
response and cold-start publication behavior unqualified.

Each summary below contains the hashes of its saved fitted-model archives;
these summary SHA-256 values pin the result against later artifact edits:

| Summary under `artifacts/` | SHA-256 |
| --- | --- |
| `heldout-causal-v1/summary.json` | `e37427b5dfafcb63b9e877413c379c786f7a89b09962493a663cffc1108800e9` |
| `heldout-causal-prior-v1/summary.json` | `c88bd4ea854add977ffd1011846a7c98c4128516a2da4e98249b8bdad3e72cf8` |
| `fresh-causal-v1/evaluation/summary.json` | `94df9752f41978d7ab4c0b3e9d4ae5be1da42f3b3411b48ce6617a72722701f3` |
| `fresh-causal-prior-v1/summary.json` | `cdbce8fc79996960e6b24eda1b7197d1e3ffd4043088aab91e6cbb395ede66dc` |
| `lag-asym-prior-full/summary.json` | `4bf602e18b4b3b5a2475da42fda12098eeced9b8033a77b8f4264d6aa467eaab` |
| `lag-asym-prior-publication/summary.json` | `8c6944f862784d151810a1027c7fec007f297c363193ff0cea74e2481cbffb9e` |
