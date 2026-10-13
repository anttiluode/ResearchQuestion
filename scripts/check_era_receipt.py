"""Compare a regenerated ERA receipt with the committed v2b summary means."""
import json
import sys
from pathlib import Path

import numpy as np

REFERENCE = Path("results/era_v2b_summary.json")
TOL = 0.02


def validate(path: str):
    actual = json.loads(Path(path).read_text())
    known = json.loads(REFERENCE.read_text())
    assert actual["apparatus_seeds"] == [41, 97]
    assert len(actual["models"]) == 16
    for world, kinds in known["means"].items():
        for kind, ref in kinds.items():
            rows = [r for r in actual["models"]
                    if r["world"] == world and r["kind"] == kind]
            assert len(rows) == 2, (world, kind)
            for metric in ("current_nrmse", "early_nrmse", "late_nrmse", "future_nrmse"):
                got = float(np.mean([r[metric] for r in rows]))
                if abs(got - ref[metric]) > TOL:
                    raise AssertionError(f"{world} {kind} {metric}: {got:.5f} vs "
                                         f"{ref[metric]:.5f}")
    print(f"ERA receipt agrees with committed means within {TOL}")


if __name__ == "__main__":
    validate(sys.argv[1])
