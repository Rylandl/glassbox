# Early control directions need excitation; longer passive fits do not replace it

The [paired early-excitation diagnostic](early-excitation-benchmark.json) uses
the sealed original arm-125 recording and the separately replayed Crazyflow
branch whose first 25 actuated commands received an orthogonal ±0.15 dither.
Each candidate is fit from rows 50–124 only, then scored against its **own**
row-125 state, next 250 ms trajectory and exact-plant 10 ms command Jacobian.
Rows 50–99 have zero issued commands in both branches. The dither changes the
row-125 state, so cross-branch trajectory errors are descriptive, not matched
treatment effects. No simulator parameter or future observation enters a fit.

| Candidate | Original response error | Dither response error | Original 250 ms rate endpoint (rad/s) | Dither endpoint (rad/s) |
| --- | ---: | ---: | ---: | ---: |
| Fast independent prompt/delayed head | 1.071 | **0.431** | **3.31** | **4.38** |
| Coupled physical-time head | 0.533 | **0.420** | 5.17 | 7.00 |
| Independent head, 0.75 s fit window | 1.064 | **0.341** | 3.19 | 3.66 |

The centered singular values of the 25 issued-command rows are
`[2.640, 0.972, 0.188, 0.122]` originally and
`[2.422, 0.967, 0.652, 0.413]` with dither. Thus the weakest/strongest ratio
improves **0.046 → 0.170**. Both independently tested heads reach roughly
0.42 relative response error with the excited data, despite their different
actuator-response structures. The independent head remains more accurate on
each branch's 250 ms rate forecast. The original/dither rows are different
physical trajectories; the comparison supports command excitation as a cause
of better local identification, not a claim of better recovery performance.

Extending the independent rate fit from its latest 25 transitions to a
physical 0.75 s window includes the 50 preceding unactuated transitions in
the original prefix. That barely changes its first response error
**1.071 → 1.064**. More passive history cannot increase command-direction
rank. On the excited branch the longer fit does use more informative
observations, improving response **0.431 → 0.341** and that branch's 250 ms
rate endpoint **4.38 → 3.66 rad/s**. Its full frozen result is mixed:

| Recording | 250 ms rate, 25-transition → 0.75 s (rad/s) | Velocity (m/s) |
| --- | ---: | ---: |
| fixedwing-80 | 0.486 → 0.486 | 0.632 → 0.632 |
| fixedwing-81 | 0.599 → 0.599 | 1.198 → 1.198 |
| quad-arm-115 | 1.165 → **0.794** | **0.269** → 0.287 |
| quad-arm-125 | **0.624** → 0.709 | **0.130** → 0.142 |
| quad-arm-135 | 6.770 → 6.698 | **0.671** → 0.764 |
| quad-change | **0.624** → 0.709 | **0.130** → 0.142 |
| paired-quad-fine | 0.004 → 0.002 | **0.042** → 0.050 |
| paired-quad-coarse | 0.001 → 0.001 | 0.007 → 0.007 |

These use all 263 origins and the same causal update sequence. `quad-change`
largely duplicates arm 125; hard arm 135 has three origins. The six-probe
arm-125 response mean worsens **0.379 → 0.693** with the longer window:
its errors are 1.064, 0.830, 1.012, 0.445, 0.208 and 0.601. Warm update
medians remain about 0.6–1.5 ms. The 0.75 s window is **not adopted** as a
generic replacement.

An exploratory per-channel prompt-fraction model also failed its smoke
screen. It selected zero prompt fraction on all four arm-125 commands, giving
the coupled model's first 250 ms error of 4.595 rad/s and first response error
of 0.533, while raising warm updates to about 11.7 ms. It was discarded
without a full run; the smoke candidate source hash was
`26c9b13e2f04c400fab8deb681a6c59cc094a97208f4c4990fe2a9c95972fafe`.
Adding per-channel lag flexibility therefore did not solve the early
recursion on the existing prefix.

The paired harness was frozen at `0533368` before fitting, and the 0.75 s
candidate was committed at `79f0bda` before its full run. Sealed artifacts in
`artifacts/readout-early-excitation-v1/` are:

| Pack | Manifest SHA-256 |
| --- | --- |
| `early-excitation-independent` | `40fd73ac8571b17e8b54d2eaf6d99b9d42d715c998bbdc3fc626fc1b02a616b9` |
| `early-excitation-coupled` | `873185c3a824b5ea3677c377fd98dd13ecb77546f2fddd43509633d487ecd593` |
| `early-excitation-long-window` | `e8f0384096515b1fde9afc63e3ca80713ed927a850bf07838903285f4a104595` |
| `long-window-rate-full` | `f4646cda041c9e229ff3e51a9a91c021786626ce8e0437e8e1a364e492e41551` |

The saved-data verifiers passed after copying each pack to the main workspace,
without refitting. This evidence separates two remaining problems: the first
command Jacobian is under-supported by the original episode, while even an
excited episode leaves a substantial 250 ms angular-recursion error. A more
constrained lag fit or a longer passive window should not be mistaken for a
solution to both. No model change was adopted into public `fit/predict/update`.
