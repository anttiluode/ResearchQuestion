# ResearchQuestion — What must a memory preserve?

> **Interactive site:** [ResearchQuestion — Future-Sufficient Memory Lab](https://anttiluode.github.io/ResearchQuestion/) — a browser-based, no-build visual explanation of when a discarded residue matters. It lets you switch a causal bridge on/off, retain four or five numbers, choose actions and questions, and compare the full system against a compact updater. The site's coefficients are illustrative and **not** the published seeded Python receipt. The recorded study remains below.


An experiment growing from the January image-mandala → Sihti → Sihti2 → SighImageFactorization → VMN / MovingTarget conversation.

> **Can a small code predict and update future responses of a much larger system without retaining every residue of the original state?**

**Yes in this deliberately constructed nonlinear test, but only when the dropped residues cannot feed back into future behavior.** The method is a positive/negative *mechanism isolator*, not automatically learned state compression or a brain/AI performance claim.

## Learned memory v2 — hidden coordinates, actual training

The next experiment is now implemented rather than proposed. See [frozen learning protocol](PROTOCOL_LEARNED.md), [trainer](learned_memory.py), and [all per-seed scores](results/learned_v2_summary.json).

**What changed:** the learning models never receive the secret 32-dimensional orthogonal basis, oracle four/five-number state, hidden coupling matrices, or privileged Jacobian probes. They see three consecutive **12-channel observations**, two preceding actions, and supervised six-query outcomes along factual training trajectories. After that short input history, they must predict **six new actions open-loop**, without seeing new sensor readings. The apparatus is a new persistent nonlinear synthetic fixture, not a natural video model and not a replay of v1's exact quotient.

Each model is trained independently on apparatus seeds 41 and 97 (1,200 train / 240 validation / 400 untouched test episodes per world and seed). All models share the same action-dependent recurrent transition and output reader, while their encoders differ:

| Six-step held-out NRMSE ↓ | No bridge feedback | Bridge feeds back |
| --- | ---: | ---: |
| Learned predictive 4-number code | 0.193 | 0.200 |
| **Learned predictive 5-number code** | **0.179** | **0.157** |
| Five-number reconstruction PCA | 0.788 | 0.782 |
| Five-state ordinary GRU encoder | 0.298 | 0.247 |

**Interpretation:** learning a task-directed observation representation improves held-out future predictions compared with PCA on raw sensor histories in this fixture. A fifth learned coordinate helps more in the bridge world, but **also helps in the closed world** (0.193 → 0.179); this does not prove it discovered the exact bridge variable. The GRU result is a baseline under one fixed learning schedule, not a hyperparameter-tuned verdict against recurrent networks. None of these models achieves an exact predictive quotient. Two apparatus seeds and supervised synthetic query labels cannot support a claim about natural images or competitive AI architecture.

**Storage:** four float32 state values = 16 bytes; five = 20 bytes *per retained episode state*. Shared encoder/update/reader parameters are additional: predictive4 **290** trainable scalars, predictive5 **381**, PCA5 **156** trained plus a frozen projection, GRU5 **501**. Short observation histories, training costs and model sharing must also be included; quoting the retained bytes alone would be misleading.

To reproduce from scratch:

```bash
pip install -r requirements-learned.txt
pytest -q tests/test_learned_memory.py
python learned_memory.py --output results/learned_memory_v2.json
```

The script writes unrounded per-episode test scores; the committed compact receipt records every seed/method/metric and paired comparisons. The existing [v1 protocol](PROTOCOL.md) remains a separate **analytically engineered** exact-state experiment.

## Classical check v2b — does the learned memory beat Ho–Kalman?

The minimal state of a system has a closed-form answer in realization theory: the rank of the map from pasts to futures (the Hankel operator; Ho–Kalman 1966, here in its switched-system form, one shift operator per action). v2b asks whether v2's gradient-trained memory beats that textbook answer. [Frozen protocol](PROTOCOL_ERA.md) committed before scoring · [code](era_baseline.py) · [summary](results/era_v2b_summary.json) · [full per-episode receipt](results/era_v2b.json).

**Method:** identical v2 data, split, 44 history features, metric and open-loop rollout. No gradient descent. Ridge-regress each action word's task response on the past (words up to length 2–3), stack the results into a generalized Hankel matrix, cut it to rank *n* by SVD, then solve one least-squares shift per action. The method is checked on an exact linear switched system: it recovers the system to < 10⁻⁶, and the singular values drop by > 10³ exactly at the true rank.

| Six-step held-out NRMSE ↓ (mean of seeds 41, 97) | Closed | Bridge |
| --- | ---: | ---: |
| Learned predictive 4 (v2) | 0.193 | 0.200 |
| Learned predictive 5 (v2) | 0.179 | **0.157** |
| **ERA 4**, closed form, frozen | 0.416 | 0.564 |
| **ERA 5**, closed form, frozen | 0.414 | 0.421 |
| ERA 4 + linear PEM refinement *(post-hoc)* | **0.153** | 0.204 |
| ERA 5 + linear PEM refinement *(post-hoc)* | 0.156 | 0.180 |

**1. Frozen verdict: the learned model wins against pure ERA in both worlds** (ERA5 − Predictive5 = +0.235 closed, +0.264 bridge; the pre-declared threshold was 0.02). The current-answer errors are similar, because both use a linear encoder on the same features (ERA5 0.28 in both worlds; Predictive5 0.28 closed, 0.23 bridge). The gap opens in the late steps: ERA's shift law is estimated from words of at most two actions and then extrapolated to six. Its validation-selected depth was 2 everywhere, because three-action words have only about 19 episodes each.

**2. The closed form is the cleaner bridge detector.** Going from 4 to 5 coordinates is worth −0.002 for ERA in the closed world and −0.143 in the bridge world. The validation rank scan shows a clear 4→5 drop only in the bridge world. For the learned models the same gains are −0.014 and −0.043. v2's open caveat was "the fifth coordinate also helps in the closed world". The Hankel rank does not show that effect, so the learned closed-world gain is more likely optimization slack than a fifth causal dimension. This is a reading, not a proof.

**3. Post-hoc diagnostic (not pre-registered).** This step refines the ERA realization by classical, strictly linear prediction-error least squares on the full six-step horizon. It uses the same 381 parameters as Predictive5, 40 solver evaluations and no validation tuning. In the closed world this linear model *beats* the learned one (0.156 vs 0.179). In the bridge world the learned tanh model keeps an edge (0.157 vs 0.180, +0.023), plausibly because the bridge couples through nonlinearities: `tanh(s₀)` drives the bridge, and the bridge enters inside `tanh`.

**What this changes:** "predictive training beats reconstruction" still stands, since PEM is also predictive. But v2's learned memory does **not** beat textbook system identification in the closed world, once that identification uses its standard refinement step. The measurable surplus of the learned model is confined to the bridge world, where nonlinearity matters. Episode-paired bootstrap comparisons against a regenerated v2 run are computed in CI ([workflow](.github/workflows/era.yml)), because the published v2 summary does not contain per-episode errors.

```bash
pip install numpy scipy pytest torch   # torch only because learned_memory.py imports it
pytest -q tests/test_era_baseline.py
python era_baseline.py --output results/era_v2b.json   # ~1 s for ERA, minutes for the PEM diagnostic
```

---

## The distinction

An encoder `m=E(X)` can be sufficient for every **current** permitted output yet insufficient for tomorrow. A **future-sufficient** representation also supports a closed update law:

```math
E(F(X,a)) = g(E(X),a).
```

For states with the same code, every allowed interaction must yield the same *next code*, and thus the same family of future observable answers. Exact reconstruction of `X` is not required. The desired property is close to a controlled invariant quotient or predictive-state sufficiency, not new physics.

## Construction

Full state: **32** coordinates, mixed into an arbitrary orthogonal physical basis. Hidden dynamical coordinates consist of **4 core values**, **1 bridge residue**, and **27 nuisance values**. Four actions update the nonlinear state; six queries read different linear combinations of its core. The state-update Jacobian has full rank 32 in both worlds.

A two-stage **telescoping projection** records core + bridge residue + nuisance residue = original state, to machine precision. This is Sihti-like conservation bookkeeping, but **not Sihti's original dyadic diffusion algorithm**.

| World | What feeds back into future queries? | Sufficient code |
| --- | --- | --- |
| **Closed** | Core, but not bridge or nuisance | 4 numbers |
| **Bridge** | Core **and one bridge residue** | 5 numbers |

The other 27 nuisance values can be discarded from the **predictive code**, although they still exist and evolve in the underlying full simulation.

## Frozen eight-seed results

The protocol was committed before the eight-seed sweep. Unit tests passed locally (11/11). All eight research gates passed in the initial local run.

| Metric | Closed world | Bridge world |
| --- | ---: | ---: |
| Full step-Jacobian rank | 32 | 32 |
| Future-response Jacobian-lens rank | 4 | 5 |
| Four-number code: future-response normalized RMSE | 3.91e-16 | **0.2737** |
| Five-number code: future-response normalized RMSE | 3.91e-16 | **4.53e-16** |
| Next-query L2 difference after perturbing bridge at fixed current answers | ~6.68e-16 | **0.8243** |
| Future-response normalized error after changing 27 nuisance values | ~8.15e-16 | ~7.94e-16 |

Eight seeds are retained in [seed_summary.csv](results/seed_summary.csv); the full-precision aggregate is [summary.json](results/summary.json). Running the script writes a **complete unrounded per-seed receipt** including Jacobian singular values to `results/receipt.json`.

**Interpretation:** in the bridge world, two states have *exactly the same answers now*, yet after an interaction their answers differ. The missing bridge residue is needed for an accurate update. In the closed world, dropping that residue is harmless for every permitted future query. Both conclusions follow from the constructed coupling graph; the experiment checks the implementations and the intended failure boundary.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q
python run_experiment.py
```

GitHub Actions reruns unit tests and all eight fixed research gates. The script reports both failing comparisons and successful controls; gates are not silently omitted.

Read [PROTOCOL.md](PROTOCOL.md) for the complete operator equations, seeds, pairing controls, predeclared thresholds and interpretation limits.

## Dimensional transfer, carefully stated

- A 32-dimensional **full-rank operator** is not a 32-number *future memory requirement*. Rank, selector-parameter count, and total storage cost are different notions.
- Four or five numbers can suffice only because **shared dynamics constrain the future tasks**. The large decoder and its costs are shared across memories but not free.
- Payload at float64 precision is 256 bytes (full), 32 bytes (core), or 40 bytes (core + bridge). These are *per-memory payloads*, **not total system savings**. The full system, encoder, decoder, measurement overhead, and model training remain unpriced.
- The diagnostic **future-response Jacobian lens** probes every physical coordinate, stacks the output derivatives after actions, and recovers the relevant 4/5-dimensional directions. This is an intentionally *privileged* analogy to downstream-effect lenses, not a claim about Anthropic's J-space results.
- A changed query family, new couplings, or richer actions might make the discarded nuisance dimensions relevant and invalidate the compressed code.
- This experiment hand-designs the invariant quotient. A meaningful *next* result would learn the quotient through limited observations and beat full-state / ordinary recurrent / fast-weight controls at **matched total cost** and held-out action sequences.

## Research ancestry (conceptual, not assumed code descent)

[2MandalaTools](https://github.com/anttiluode/2MandalaTools) generates elaborate patterns through shared rules; [Sihti](https://github.com/anttiluode/Sihti) keeps exact disappearing components; [Sihti2](https://github.com/anttiluode/Sihti2) studies geometry written by data; [SighImageFactorization](https://github.com/anttiluode/SighImageFactorization) tests history-written grouping operators; [VMN](https://github.com/anttiluode/VMN) and [VMNClaude](https://github.com/anttiluode/VMNClaude) study state-conditioned responses and query costs; [MovingTarget2](https://github.com/anttiluode/MovingTarget2) shows that an answer code need not update itself under subsequent reads. [Anthropic's J-space research](https://transformer-circuits.pub/2026/workspace/index.html) motivates measuring representations by downstream effects but does not establish the memory construction tested here.

**The question:** what does a representation need to preserve so that the *next* computation remains correct?
