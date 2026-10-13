"""Episode-paired comparison of ERA rows against a regenerated v2 learned run.

Both receipts must come from the same fixed seeds and test split. Positive
mean difference = ERA row has larger error than the learned model.
"""
import json
import sys
from pathlib import Path

import numpy as np

PAIRS = [("era5", "predictive5"), ("era4", "predictive4"),
         ("era5_pem_posthoc", "predictive5"), ("era4_pem_posthoc", "predictive4")]


def compare(learned_path: str, era_path: str, out_path: str):
    learned = {(r["seed"], r["world"], r["kind"]): r
               for r in json.loads(Path(learned_path).read_text())["models"]}
    era = {(r["seed"], r["world"], r["kind"]): r
           for r in json.loads(Path(era_path).read_text())["models"]}
    rng = np.random.default_rng(2026)
    report = {}
    for world in ("closed", "bridge"):
        report[world] = {}
        for a, b in PAIRS:
            diff = np.concatenate([
                np.array(era[(s, world, a)]["paired_test_episode_nrmse"]) -
                np.array(learned[(s, world, b)]["paired_test_episode_nrmse"])
                for s in (41, 97)])
            boot = [rng.choice(diff, diff.size).mean() for _ in range(4000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            report[world][f"{a}_vs_{b}"] = {
                "episodes": int(diff.size),
                "mean_episode_difference": round(float(diff.mean()), 5),
                "bootstrap_95ci": [round(float(lo), 5), round(float(hi), 5)],
                "era_wins_fraction": round(float((diff < 0).mean()), 4)}
            print(world, a, "vs", b, report[world][f"{a}_vs_{b}"], flush=True)
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    compare(*sys.argv[1:4])
