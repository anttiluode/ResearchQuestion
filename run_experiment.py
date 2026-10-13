"""Predeclared fixed-seed experiment. Writes a complete machine-readable receipt."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from research_question import (make_world, random_full, local_future_lens,
                               lens_dimension, projection_distance,
                               full_trajectory, query_trajectory, normed_rmse,
                               physical_step_jacobian)

SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
STEPS = 12
LEAK = 0.28


def run_seed(seed: int, leak: float) -> dict:
    w = make_world(seed=seed, leak=leak)
    rng = np.random.default_rng(seed + 7_000_000)
    contexts = np.stack([random_full(w, rng) for _ in range(3)])
    singular, vt = local_future_lens(w, contexts)
    rank = lens_dimension(singular)
    k = w.core_dim
    truth_basis = w.q[:, :k + int(leak != 0)].T
    lens_err = projection_distance(vt[:rank], truth_basis)
    full_jac = physical_step_jacobian(w, contexts[0], 0)
    ambient_jacobian_rank = int(np.linalg.matrix_rank(full_jac, tol=1e-7))

    missing_errs, enough_errs, full_sieve_errors = [], [], []
    state_pair_differences, inert_pair_differences = [], []
    for _ in range(24):
        x = random_full(w, rng)
        actions = rng.integers(0, w.action_count, size=STEPS).tolist()
        true = full_trajectory(w, x, actions)
        missing_errs.append(normed_rmse(query_trajectory(w, x, actions, False), true))
        enough_errs.append(normed_rmse(query_trajectory(w, x, actions, True), true))
        core, bridge, nuisance = w.sieve(x)
        full_sieve_errors.append(np.linalg.norm(x - (core+bridge+nuisance)) /
                                 max(np.linalg.norm(x), 1e-12))
        x_bridge_twin = x + 1.3 * w.q[:, k]
        y1 = np.array([w.read(w.step(x, a), q) for a in range(w.action_count)
                       for q in range(w.query_count)])
        y2 = np.array([w.read(w.step(x_bridge_twin, a), q)
                       for a in range(w.action_count)
                       for q in range(w.query_count)])
        state_pair_differences.append(float(np.linalg.norm(y1-y2)))
        x_nuisance_twin = x + w.q[:, k+1:] @ rng.normal(size=w.dimension-k-1)
        inert_pair_differences.append(normed_rmse(
            full_trajectory(w, x_nuisance_twin, actions), true))
    return dict(seed=seed, dimension=w.dimension, core_dim=k,
                discovered_lens_rank=rank,
                ambient_step_jacobian_rank=ambient_jacobian_rank,
                lens_singular_values=[float(x) for x in singular[:8]],
                lens_projector_error=lens_err,
                core_only_future_nrmse=float(np.mean(missing_errs)),
                core_plus_bridge_future_nrmse=float(np.mean(enough_errs)),
                exact_sieve_relative_error=max(full_sieve_errors),
                changed_bridge_next_response_l2=float(np.mean(state_pair_differences)),
                changed_nuisance_future_nrmse=float(np.mean(inert_pair_differences)))


def describe_leak(leak: float) -> dict:
    cases = [run_seed(seed, leak) for seed in SEEDS]
    metrics = ["ambient_step_jacobian_rank", "discovered_lens_rank",
               "lens_projector_error", "core_only_future_nrmse",
               "core_plus_bridge_future_nrmse", "exact_sieve_relative_error",
               "changed_bridge_next_response_l2",
               "changed_nuisance_future_nrmse"]
    aggregate = {m: {"mean": float(np.mean([r[m] for r in cases])),
                     "max": float(np.max([r[m] for r in cases]))} for m in metrics}
    return {"leak": leak, "by_seed": cases, "aggregate": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/receipt.json")
    args = parser.parse_args()
    report = {"study": "exact residue versus sufficient future state",
              "seed_list": SEEDS, "steps": STEPS, "coupled_leak": LEAK,
              "closed": describe_leak(0.0), "bridge_needed": describe_leak(LEAK),
              "scope": "Constructed nonlinear system; privileged full-coordinate finite-difference lens",
              "precision": "float64",
              "per_memory_payload_bytes": {"full": 32*8, "core": 4*8,
                                           "core_and_bridge": 5*8},
              "shared_cost_not_counted_in_payload": True}
    c, l = report["closed"]["aggregate"], report["bridge_needed"]["aggregate"]
    report["gates"] = {
        "G0_exact_sieve": max(c["exact_sieve_relative_error"]["max"],
                              l["exact_sieve_relative_error"]["max"]) < 1e-12,
        "G1_closed_core": c["core_only_future_nrmse"]["max"] < 1e-10,
        "G2_leak_breaks_core": l["core_only_future_nrmse"]["mean"] > 0.02,
        "G3_bridge_restores_closure": l["core_plus_bridge_future_nrmse"]["max"] < 1e-10,
        "G4_nuisance_can_go": l["changed_nuisance_future_nrmse"]["max"] < 1e-10,
        "G5_lens_recovers_ranks": c["discovered_lens_rank"]["mean"] == 4 and
                                    l["discovered_lens_rank"]["mean"] == 5 and
                                    max(c["lens_projector_error"]["max"],
                                        l["lens_projector_error"]["max"]) < 1e-6,
        "G6_bridge_pair_diverges": l["changed_bridge_next_response_l2"]["mean"] > 0.05,
        "G7_full_rank_law_small_future_code": c["ambient_step_jacobian_rank"]["mean"] == 32 and
                                              l["ambient_step_jacobian_rank"]["mean"] == 32,
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    for worldname in ("closed", "bridge_needed"):
        row = report[worldname]
        print(f"{worldname} (leak={row['leak']})")
        for name in ("ambient_step_jacobian_rank", "discovered_lens_rank",
                     "lens_projector_error", "core_only_future_nrmse",
                     "core_plus_bridge_future_nrmse", "changed_bridge_next_response_l2",
                     "changed_nuisance_future_nrmse"):
            print(f"  {name}: mean={row['aggregate'][name]['mean']:.9g} "
                  f"max={row['aggregate'][name]['max']:.9g}")
    print("GATES", report["gates"])
    if not all(report["gates"].values()):
        raise SystemExit("At least one preregistered gate failed; inspect receipt")


if __name__ == "__main__":
    main()
