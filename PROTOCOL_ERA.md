# Frozen v2b protocol — classical Ho–Kalman / ERA baseline for the learned memory

Fixed on 2026-10-13, **before** any ERA test score was computed, on top of commit `082d46d` (v2 as published). This adds a baseline; it changes nothing in v2's data, split, models or published numbers.

## Question

ResearchQuestion v2 trained small predictive memories by gradient descent. Classical realization theory (Ho–Kalman 1966; its switched-system form, e.g. Petreczky) gives the minimal state *in closed form*: the rank of the map from pasts to word-conditioned futures. Does the learned model beat the closed-form linear answer, tie it, or lose to it?

## Data and metric — identical to v2

`learned_memory.build_splits(seed, leak)` for apparatus seeds **41, 97**, closed (`leak=0`) and bridge (`leak=0.28`) worlds; same 1,200 / 240 / 400 split; same three-frame sensor history plus two past actions (44 features) standardized with training-only statistics; same six task queries; same open-loop six-step rollout; same metric (pooled RMSE over steps 1–6 divided by RMS of true outputs at steps 1–6, plus current / early 1–3 / late 4–6). The learner never sees the hidden state, the rotation, the sensor matrix, the partition or the update law.

## Method — switched-system ERA, no gradient descent

1. **Generalized Hankel blocks.** For every action word `w` of length `0..L`, take the training episodes whose future action string begins with `w` and ridge-regress the observed task response `y_|w|` on the past features: `y ≈ M_w h + c_w`. (Ridge on `M_w` only.)
2. **Rank cut.** Stack the `M_w` into `G` (rows weighted by `sqrt(episode count)`). Truncated SVD to rank `n` gives `G ≈ O K`. Encoder `x0 = K h` (n floats per episode). The constant responses `c_w` become one extra *fixed* state coordinate equal to 1 (not stored per episode).
3. **Shift invariance.** For each action `a`, solve `O_(a·u) = O_u Ã_a` by weighted least squares over all words `u` of length `≤ L−1`; the constant coordinate is held at 1. `C̃ = O_(empty word)`.
4. **Open-loop rollout.** `x̃_{t+1} = Ã_{a_t} x̃_t`, `y_t = C̃ x̃_t`, linear, no further sensor readings.

Primary models: **ERA4** (`n = 4`) and **ERA5** (`n = 5`). The only selectable settings are word depth `L ∈ {2, 3}` and ridge `λ ∈ {0.01, 0.1, 1, 10, 100}`, chosen per seed/world/model by **validation** MSE over steps 0–6 (the same criterion v2 used for checkpoints). Words with fewer than 8 training episodes are dropped. No test-set tuning. A validation-only rank scan `n = 1..8` is reported as a diagnostic.

## Pre-declared reading

Compare against the published v2 means (Predictive5: closed 0.1785, bridge 0.1571; Predictive4: closed 0.1927, bridge 0.2000).

- **ERA5 within 0.01 of Predictive5, or better:** the learned model's advantage over the classical closed-form quotient is *not* demonstrated in that world. v2 is then rediscovering the textbook answer, and the README must say so.
- **ERA5 worse than Predictive5 by more than 0.02:** the learned model's nonlinear updater/training adds value beyond the linear Hankel quotient in that world.
- **Otherwise:** a tie at this precision.
- **Bridge question:** is ERA5 − ERA4 larger in the bridge world than in the closed world? If yes, the closed form sees the bridge dimension too.

All outcomes are published, including any in which ERA wins outright.

## Known limitations, stated in advance

The fixture is nonlinear (`tanh`, `sin`), so a linear switched realization is misspecified; that is intended — it is what the classical method can do without learning. Long words have few episodes, so their Hankel rows are noisy; the `sqrt(count)` weighting and the validation choice of `L` are the only defences. Storage accounting: per-episode retained state is `n` float32s for every model; the shared encoder/operators are counted separately.
