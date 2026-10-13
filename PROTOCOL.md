# Frozen protocol — future-sufficient code versus exact residue

Design fixed on **13 October 2026**, before the eight-seed outcome sweep. This is a constructed mathematical counterexample and positive control, not the spontaneous discovery of a biological or image representation.

## Research question

Must Sihti-like residues be kept after dimensional collapse? Only if they can change a future answer or the future compressed state under permitted interactions. For `m=E(X)`, a closed predictive code obeys

```math
E(F(X,a)) = g(E(X),a)
```

for all relevant full states `X` and actions `a`; allowed outputs must depend only on `m`.

## Synthetic apparatus

The full system has **32 float64 coordinates**. A randomly generated orthogonal matrix `Q` hides coordinates `z=Q.T@X=[s_0,...,s_3, b, n_0,...,n_26]`. There are **4 actions**, **6 queries**, and a deliberately nonlinear dynamical law:

```
s' = tanh(A[a] @ s + 0.13 sin(s) + leak * v[a] * b)
b' = rho[a] * b + 0.11 tanh(s[0])
n' = tanh(H[a] @ n + 0.06 d[a] * b)
answer(X,q) = c[q] @ s
```

Queries are nondestructive. State updates happen after an action. The `n` coordinates never feed back to `b` or `s`, by construction.

- **Closed world:** `leak=0`; the four-number core `s` is closed under future actions, even though `b,n` physically persist.
- **Bridge world:** `leak=0.28`; `s` is sufficient for *current* answers, but the next `s` needs one additional residual coordinate `b`. The five-number code `(s,b)` is closed; all 27 `n` coordinates remain dispensable relative to this particular task.

## Exact sieve (not Sihti's particular purifier)

Nested orthogonal projections form a two-step telescoping ledger:

```
core       = Q[:, :4] @ z[:4]
R_bridge   = Q[:, 4] * z[4]
R_nuisance = X - core - R_bridge
X = core + R_bridge + R_nuisance
```

This is structurally analogous to Sihti's exact bookkeeping, not a replication of its dyadic diffusion. Keeping all residues guarantees full input reconstruction; discarding a residue only preserves the specified future behavior when that residue is causally irrelevant to it.

## Frozen measurement protocol

Use eight seeds **1001–1008**, 24 independent initial states per seed, 12 random actions per trajectory. Measure full-simulator six-query responses at every time and compare against compact-code updates *without reopening the full discarded state*. Report per-seed statistics and aggregate means/maxima.

Controls:
1. Exact telescope reconstruction error.
2. Four- versus five-number predictive code errors under matched action sequences.
3. Same-current-answer pairs: keep `s` fixed, perturb `b`, and compare next answers.
4. Same-`s,b` pairs: perturb `n`, and compare entire future output trajectories.
5. Privileged Jacobian lens: finite-difference every full-state coordinate in three contexts, over all actions and all subsequent query types; SVD the stacked derivatives and compare the discovered 4/5-dimensional subspaces to their known true projections.
6. Rank of the full 32x32 state-update Jacobian. This separates rank of the machinery from dimension of a task-sufficient memory.

## Prespecified gates

- **G0**: max relative exact-sieve reconstruction error < 1e-12 in both worlds.
- **G1**: closed-world four-number future-response NRMSE < 1e-10 (max across seeds).
- **G2**: bridge-world four-number NRMSE > 0.02 (mean across seeds; missing bridge estimated as zero).
- **G3**: bridge-world five-number NRMSE < 1e-10 (max across seeds).
- **G4**: the 27 nuisance axes can be changed without changing future answers, NRMSE < 1e-10.
- **G5**: future-response Jacobian-lens ranks are 4 and 5, respectively; projector error < 1e-6.
- **G6**: same-answer states separated along the bridge diverge after one action by mean L2 response distance > 0.05.
- **G7**: rank of the full ambient step Jacobian remains 32 in both worlds.

## Boundaries and failure modes

This world was designed to have a closed quotient: passing the gates demonstrates a mechanism, **not** learned compression, performance in natural image or brain systems, or new physical mathematics. The lens is an expensive privileged measurement, not a cheap sensor or decoder. The model and `Q` are shared costs that must be counted separately from each additional stored memory; the physical full state is retained by the simulator.

The quotient is relative to the allowed queries and actions. Permit direct queries of `n`, or let `n` feed back into `s,b`, and the supposed redundant residue may become necessary. The natural next challenge is acquiring a useful quotient using limited observations and learned dynamics rather than engineered access to `Q`.
