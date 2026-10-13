"""Classical baseline for v2: switched-system Ho-Kalman / ERA in closed form.

Frozen in PROTOCOL_ERA.md before scoring. Uses the identical v2 data, split,
history features and metric from learned_memory.py. No gradient descent:
ridge regressions to word-conditioned responses, one SVD, one least-squares
shift per action, then a linear open-loop rollout.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from learned_memory import (ACTIONS, APPARATUS_SEEDS, HISTORY, HORIZON,
                            QUERIES, SENSORS, Episodes, build_splits)

DEPTHS = (2, 3)
RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0)
MIN_COUNT = 8
PRIMARY_RANKS = (4, 5)
SCAN_RANKS = tuple(range(1, 9))


def standardize(train: Episodes):
    dim = SENSORS * HISTORY
    mean = train.history[:, :dim].mean(axis=0)
    std = np.maximum(train.history[:, :dim].std(axis=0), 1e-4)

    def features(ep: Episodes) -> np.ndarray:
        x = ep.history.copy()
        x[:, :dim] = (x[:, :dim] - mean) / std
        return x
    return features


def words_upto(depth: int):
    for length in range(depth + 1):
        yield from itertools.product(range(ACTIONS), repeat=length)


def hankel_blocks(h: np.ndarray, ep: Episodes, depth: int, ridge: float):
    """Ridge-regress each word's response on the past: y = M_w h + c_w."""
    blocks = {}
    for word in words_upto(depth):
        mask = np.all(ep.future_actions[:, :len(word)] == np.array(word, int),
                      axis=1) if word else np.ones(len(h), bool)
        count = int(mask.sum())
        if count < MIN_COUNT:
            continue
        x, y = h[mask], ep.targets[mask, len(word)]
        xm, ym = x.mean(axis=0), y.mean(axis=0)
        xc, yc = x - xm, y - ym
        m = np.linalg.solve(xc.T @ xc + ridge * np.eye(x.shape[1]), xc.T @ yc).T
        blocks[word] = (m, ym - m @ xm, count)
    return blocks


def realize(blocks: dict, depth: int, rank: int) -> dict:
    """Weighted SVD rank cut + shift invariance -> (K, A_a, C)."""
    words = list(blocks)
    rows = {w: slice(i * QUERIES, (i + 1) * QUERIES) for i, w in enumerate(words)}
    g = np.vstack([blocks[w][0] for w in words])
    c = np.concatenate([blocks[w][1] for w in words])
    wt = np.concatenate([np.full(QUERIES, np.sqrt(blocks[w][2])) for w in words])
    u, s, vt = np.linalg.svd(wt[:, None] * g, full_matrices=False)
    r = min(rank, len(s))
    obs = (u[:, :r] * np.sqrt(s[:r])) / wt[:, None]
    enc = np.sqrt(s[:r])[:, None] * vt[:r]
    full = np.hstack([obs, c[:, None]])          # constant coordinate = 1
    trans = []
    for a in range(ACTIONS):
        base, shifted, w8 = [], [], []
        for w in words:
            if len(w) <= depth - 1 and (a,) + w in rows:
                base.append(full[rows[w]])
                shifted.append(full[rows[(a,) + w]])
                w8.append(wt[rows[(a,) + w]])
        base, shifted, w8 = np.vstack(base), np.vstack(shifted), np.concatenate(w8)
        rhs = shifted - base[:, r:] @ np.eye(r + 1)[r:]   # constant row fixed
        top, *_ = np.linalg.lstsq(w8[:, None] * base[:, :r], w8[:, None] * rhs,
                                  rcond=None)
        trans.append(np.vstack([top, np.eye(r + 1)[r:]]))
    return {"encoder": enc, "trans": np.stack(trans), "reader": full[rows[()]],
            "singular_values": s, "rank": r}


def rollout(model: dict, h: np.ndarray, actions: np.ndarray) -> np.ndarray:
    x0 = h @ model["encoder"].T + model.get("bias", 0.0)
    x = np.hstack([x0, np.ones((len(h), 1))])
    out = [x @ model["reader"].T]
    for t in range(HORIZON):
        x = np.einsum("bij,bj->bi", model["trans"][actions[:, t]], x)
        out.append(x @ model["reader"].T)
    return np.stack(out, axis=1)


def refine_pem(model: dict, h: np.ndarray, ep: Episodes, max_nfev=40) -> dict:
    """POST-HOC diagnostic, not part of the frozen primary: classical
    prediction-error refinement of the ERA realization on the full six-step
    horizon. Stays strictly linear; adds an encoder bias so the parameter
    count matches Predictive5 exactly (n=5 -> 381)."""
    from scipy.optimize import least_squares
    n = model["rank"]
    shapes = [("encoder", (n, h.shape[1])), ("bias", (n,)),
              ("top", (ACTIONS, n, n + 1)), ("reader", (QUERIES, n + 1))]
    init = np.concatenate([model["encoder"].ravel(), np.zeros(n),
                           model["trans"][:, :n].ravel(), model["reader"].ravel()])
    scale = float(np.sqrt(np.mean(ep.targets ** 2)))

    def unpack(theta):
        out, i = {}, 0
        for name, shp in shapes:
            size = int(np.prod(shp))
            out[name] = theta[i:i + size].reshape(shp)
            i += size
        fixed = np.broadcast_to(np.eye(n + 1)[n:], (ACTIONS, 1, n + 1))
        return {"encoder": out["encoder"], "bias": out["bias"],
                "trans": np.concatenate([out["top"], fixed], axis=1),
                "reader": out["reader"], "rank": n}

    def resid(theta):
        return ((rollout(unpack(theta), h, ep.future_actions) - ep.targets)
                / scale).ravel()
    fit = least_squares(resid, init, method="trf", x_scale="jac",
                        max_nfev=max_nfev)
    refined = unpack(fit.x)
    refined["parameters"] = int(fit.x.size)
    return refined


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    energy = max(float(np.sqrt(np.mean(truth[:, 1:] ** 2))), 1e-12)

    def err(s):
        return float(np.sqrt(np.mean((pred[:, s] - truth[:, s]) ** 2)) / energy)
    ep = np.sqrt(np.mean((pred[:, 1:] - truth[:, 1:]) ** 2, axis=(1, 2))) / energy
    return {"current_nrmse": err(slice(0, 1)), "early_nrmse": err(slice(1, 4)),
            "late_nrmse": err(slice(4, 7)), "future_nrmse": err(slice(1, 7)),
            "paired_test_episode_nrmse": ep.tolist()}


def fit_world(seed: int, leak: float) -> list[dict]:
    data = build_splits(seed, leak)
    feat = standardize(data["train"])
    ht, hv, he = (feat(data[k]) for k in ("train", "val", "test"))
    cache = {(d, lam): hankel_blocks(ht, data["train"], d, lam)
             for d in DEPTHS for lam in RIDGES}

    def val_mse(model):
        p = rollout(model, hv, data["val"].future_actions)
        return float(np.mean((p - data["val"].targets) ** 2))

    def best(rank):
        options = [(val_mse(m := realize(b, d, rank)), d, lam, m)
                   for (d, lam), b in cache.items()]
        return min(options, key=lambda o: o[0])

    world = "bridge" if leak else "closed"
    scan = {n: best(n)[0] for n in SCAN_RANKS}
    rows = []
    for n in PRIMARY_RANKS:
        vmse, depth, lam, model = best(n)
        pred = rollout(model, he, data["test"].future_actions)
        item = {"kind": f"era{n}", "seed": seed, "world": world,
                "retained_state": n, "marginal_state_bytes_float32": 4 * n,
                "shared_parameters": int(model["encoder"].size +
                                         model["trans"][:, :n].size +
                                         model["reader"].size),
                "word_depth": depth, "ridge": lam,
                "hankel_words": len(cache[(depth, lam)]),
                "val_mse": vmse,
                "train_hankel_singular_values": model["singular_values"][:10].tolist(),
                "validation_rank_scan_mse": {str(k): v for k, v in scan.items()},
                **scores(pred, data["test"].targets)}
        print(f"{seed:3d} {world:6s} era{n}: now={item['current_nrmse']:.3f} "
              f"early={item['early_nrmse']:.3f} late={item['late_nrmse']:.3f} "
              f"future={item['future_nrmse']:.3f}  (L={depth}, ridge={lam})",
              flush=True)
        rows.append(item)
        # Post-hoc diagnostic (labelled; not a frozen primary model).
        pem = refine_pem(model, ht, data["train"])
        pv = rollout(pem, hv, data["val"].future_actions)
        diag = {"kind": f"era{n}_pem_posthoc", "seed": seed, "world": world,
                "retained_state": n, "shared_parameters": pem["parameters"],
                "linear": True, "posthoc_diagnostic": True,
                "val_mse": float(np.mean((pv - data["val"].targets) ** 2)),
                **scores(rollout(pem, he, data["test"].future_actions),
                         data["test"].targets)}
        print(f"{seed:3d} {world:6s} era{n}+PEM (post-hoc, linear): "
              f"now={diag['current_nrmse']:.3f} early={diag['early_nrmse']:.3f} "
              f"late={diag['late_nrmse']:.3f} future={diag['future_nrmse']:.3f}",
              flush=True)
        rows.append(diag)
    return rows


def run(seeds=APPARATUS_SEEDS) -> dict:
    rows = [r for seed in seeds for leak in (0.0, 0.28)
            for r in fit_world(seed, leak)]
    return {"protocol": "PROTOCOL_ERA.md", "apparatus_seeds": list(seeds),
            "method": "switched-system Ho-Kalman/ERA, closed form, linear rollout",
            "depths": list(DEPTHS), "ridges": list(RIDGES),
            "min_word_count": MIN_COUNT, "models": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/era_v2b.json")
    args = parser.parse_args()
    report = run()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n",
                                 encoding="utf-8")


if __name__ == "__main__":
    main()
