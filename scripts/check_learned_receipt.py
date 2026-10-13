"""Compare regenerated v2 receipt with the fixed-seed publication.

Tolerance accommodates small library/platform numerical differences; this
is NOT a gate that assumes a particular learner must outperform a baseline.
"""
import json
import sys
from pathlib import Path

REFERENCE = Path("results/learned_v2_summary.json")
TOL = 0.025


def validate(path: str):
    actual = json.loads(Path(path).read_text())
    known = json.loads(REFERENCE.read_text())
    assert actual["apparatus_seeds"] == known["apparatus_seeds"]
    assert actual["epochs"] == known["epochs"]
    assert len(actual["models"]) == len(known["runs"]) == 16
    dictionary = {(r["seed"], r["world"], r["kind"]): r
                  for r in actual["models"]}
    for r in known["runs"]:
        key = (r["seed"], r["world"], r["method"])
        case = dictionary[key]
        assert len(case["paired_test_episode_nrmse"]) == known["test_episodes"]
        for metric in ("current_nrmse", "early_nrmse", "late_nrmse", "future_nrmse"):
            diff = abs(case[metric] - r[metric])
            if diff > TOL:
                raise AssertionError(f"Regeneration mismatch {key}, {metric}: "
                                     f"{case[metric]:.5g} vs {r[metric]:.5g}, "
                                     f"absolute delta={diff:.5g}")
    print("All 16 fixed-seed runs agree with the published scores "
          f"within {TOL} absolute normalized error")


if __name__ == "__main__":
    validate(sys.argv[1])
