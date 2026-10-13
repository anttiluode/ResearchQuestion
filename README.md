# ResearchQuestion — What must a memory preserve?

An experiment growing from the January image-mandala → Sihti → Sihti2 → SighImageFactorization → VMN / MovingTarget conversation.

> **Can a small code predict and update future responses of a much larger system without retaining every residue of the original state?**

**Yes in this deliberately constructed nonlinear test, but only when the dropped residues cannot feed back into future behavior.** The method is a positive/negative *mechanism isolator*, not automatically learned state compression or a brain/AI performance claim.

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
