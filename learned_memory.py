"""Learn compact predictive memories from partial sensor histories.

The simulator alone knows the 32D hidden coordinates. Models only see twelve
observation channels, two prior actions, future action strings and supervised
six-query outcomes. Separate synthetic fixture from the exact v1 theorem.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

SENSORS = 12
HISTORY = 3
ACTIONS = 4
QUERIES = 6
HORIZON = 6
INPUT_DIM = SENSORS * HISTORY + 2 * ACTIONS
APPARATUS_SEEDS = (41, 97)
SPLIT_COUNTS = {"train": 1200, "val": 240, "test": 400}
MODEL_NAMES = ("predictive4", "predictive5", "pca5", "gru5")


def make_apparatus(seed: int, leak: float) -> dict:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(32, 32)))
    q *= np.sign(np.diag(r))
    sensor = rng.normal(size=(SENSORS, 32)) / np.sqrt(32)
    # No privileged selection of the five relevant hidden directions.
    observed = sensor @ q
    a = np.stack([0.80 * np.eye(4) + rng.normal(scale=0.085, size=(4, 4))
                  for _ in range(ACTIONS)])
    bdrive = rng.normal(size=(ACTIONS, 4))
    bdrive /= np.linalg.norm(bdrive, axis=1, keepdims=True)
    reads = rng.normal(size=(QUERIES, 4))
    reads /= np.linalg.norm(reads, axis=1, keepdims=True)
    rho = rng.uniform(.88, .96, size=ACTIONS)
    bforce = rng.normal(scale=.04, size=ACTIONS)
    ndrive = rng.normal(size=(ACTIONS, 27))
    return {"sensor": observed, "a": a, "bdrive": bdrive,
            "reads": reads, "rho": rho, "bforce": bforce,
            "ndrive": ndrive, "leak": leak}


def evolve(z: np.ndarray, action: np.ndarray, app: dict) -> np.ndarray:
    """Hidden-state law: simulator only, never invoked by learned models."""
    s, b, n = z[:, :4], z[:, 4], z[:, 5:]
    mixed = np.einsum("bij,bj->bi", app["a"][action], s)
    sp = np.tanh(mixed + .09 * np.sin(s) +
                 app["leak"] * app["bdrive"][action] * b[:, None])
    bp = app["rho"][action] * b + .07 * np.tanh(s[:, 0]) + app["bforce"][action]
    npart = np.tanh(.40 * n + .06 * np.sin(n) +
                     .035 * app["ndrive"][action] * b[:, None])
    return np.concatenate([sp, bp[:, None], npart], axis=1)


@dataclass
class Episodes:
    history: np.ndarray
    sequence: np.ndarray
    future_actions: np.ndarray
    targets: np.ndarray


def generate(app: dict, seed: int, amount: int) -> Episodes:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(amount, 32)) * np.array([.67] * 4 + [.85] + [.8] * 27)
    observations = [z @ app["sensor"].T]
    past = rng.integers(0, ACTIONS, size=(amount, HISTORY - 1))
    for j in range(HISTORY - 1):
        z = evolve(z, past[:, j], app)
        observations.append(z @ app["sensor"].T +
                            rng.normal(scale=.002, size=(amount, SENSORS)))
    future = rng.integers(0, ACTIONS, size=(amount, HORIZON))
    targets = [z[:, :4] @ app["reads"].T]
    for j in range(HORIZON):
        z = evolve(z, future[:, j], app)
        targets.append(z[:, :4] @ app["reads"].T)
    history = np.concatenate(observations + [
        np.eye(ACTIONS)[past[:, j]] for j in range(HISTORY - 1)], axis=1)
    seq = np.concatenate([np.stack(observations, axis=1),
        np.stack([np.zeros((amount, ACTIONS)), np.eye(ACTIONS)[past[:, 0]],
                  np.eye(ACTIONS)[past[:, 1]]], axis=1)], axis=2)
    return Episodes(history, seq, future, np.stack(targets, axis=1))


def build_splits(seed: int, leak: float) -> dict[str, Episodes]:
    app = make_apparatus(seed, leak)
    split_seeds = {"train": 11, "val": 23, "test": 47}
    return {name: generate(app, seed * 1_000_000 + split_seeds[name], n)
            for name, n in SPLIT_COUNTS.items()}


class Standardizer:
    def __init__(self, training: Episodes):
        dim = SENSORS * HISTORY
        self.x_mean = training.history[:, :dim].mean(axis=0)
        self.x_std = np.maximum(training.history[:, :dim].std(axis=0), 1e-4)
        self.y_scale = max(float(np.sqrt(np.mean(training.targets**2))), 1e-5)

    def arrays(self, ep: Episodes):
        x = ep.history.copy()
        dim = SENSORS * HISTORY
        x[:, :dim] = (x[:, :dim] - self.x_mean) / self.x_std
        seq = ep.sequence.copy()
        for i in range(HISTORY):
            part = slice(i * SENSORS, (i + 1) * SENSORS)
            seq[:, i, :SENSORS] = (
                seq[:, i, :SENSORS] - self.x_mean[part]) / self.x_std[part]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(seq, dtype=torch.float32),
                torch.tensor(ep.future_actions, dtype=torch.long),
                torch.tensor(ep.targets / self.y_scale, dtype=torch.float32))


class Predictor(nn.Module):
    def __init__(self, kind: str,
                 pca: tuple[np.ndarray, np.ndarray] | None = None):
        super().__init__()
        self.kind = kind
        self.dim = 4 if kind == "predictive4" else 5
        if kind in ("predictive4", "predictive5"):
            self.encoder = nn.Linear(INPUT_DIM, self.dim)
        elif kind == "pca5":
            assert pca is not None
            self.register_buffer("pca_mean",
                                 torch.tensor(pca[0], dtype=torch.float32))
            self.register_buffer("pca_components",
                                 torch.tensor(pca[1], dtype=torch.float32))
        elif kind == "gru5":
            self.encoder = nn.GRU(SENSORS + ACTIONS, self.dim,
                                  batch_first=True)
        else:
            raise ValueError(kind)
        self.trans = nn.Parameter(torch.eye(self.dim).repeat(ACTIONS, 1, 1))
        self.bias = nn.Parameter(torch.zeros(ACTIONS, self.dim))
        self.reader = nn.Linear(self.dim, QUERIES)

    def encode(self, x, seq):
        if self.kind == "pca5":
            return (x - self.pca_mean) @ self.pca_components.T
        if self.kind == "gru5":
            return self.encoder(seq)[1][-1]
        return self.encoder(x)

    def forward(self, x, seq, actions):
        state = self.encode(x, seq)
        answers = [self.reader(state)]
        for t in range(HORIZON):
            state = torch.tanh(
                torch.bmm(state[:, None, :],
                          self.trans[actions[:, t]]).squeeze(1)
                + self.bias[actions[:, t]])
            answers.append(self.reader(state))
        return torch.stack(answers, dim=1)


def train_one(train: Episodes, valid: Episodes, test: Episodes, kind: str,
              apparatus_seed: int, leak: float, epochs=160,
              batch_size=300) -> dict:
    torch.manual_seed(20000 + apparatus_seed * 7 + MODEL_NAMES.index(kind))
    scaler = Standardizer(train)
    tx, ts, ta, ty = scaler.arrays(train)
    vx, vs, va, vy = scaler.arrays(valid)
    ex, es, ea, _ = scaler.arrays(test)
    pca = None
    if kind == "pca5":
        center = tx.numpy().mean(axis=0)
        _, _, vt = np.linalg.svd(tx.numpy() - center, full_matrices=False)
        pca = (center, vt[:5].copy())
    model = Predictor(kind, pca=pca)
    optimizer = torch.optim.Adam(model.parameters(), lr=.006)
    gen = torch.Generator().manual_seed(
        apparatus_seed * 100 + MODEL_NAMES.index(kind) + 1000)
    best = float("inf")
    best_state = None
    checkpoint = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(tx), generator=gen)
        for idx in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            pred = model(tx[idx], ts[idx], ta[idx])
            loss = torch.mean((pred - ty[idx]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val = float(torch.mean(
                    (model(vx, vs, va) - vy) ** 2))
            if val < best:
                best = val
                best_state = copy.deepcopy(model.state_dict())
                checkpoint = epoch
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(ex, es, ea).numpy() * scaler.y_scale
    truth = test.targets
    energy = max(float(np.sqrt(np.mean(truth[:, 1:] ** 2))), 1e-12)

    def error(s):
        return float(np.sqrt(np.mean(
            (pred[:, s] - truth[:, s]) ** 2)) / energy)

    ep_err = np.sqrt(np.mean(
        (pred[:, 1:] - truth[:, 1:]) ** 2, axis=(1, 2))) / energy
    return {"kind": kind, "seed": apparatus_seed,
            "world": "bridge" if leak else "closed",
            "retained_state": model.dim,
            "marginal_state_bytes_float32": model.dim * 4,
            "shared_trainable_parameters":
                sum(p.numel() for p in model.parameters()),
            "shared_frozen_parameters": INPUT_DIM * 5 + INPUT_DIM
                if kind == "pca5" else 0,
            "checkpoint_epoch": checkpoint,
            "val_mse_normalized": best,
            "current_nrmse": error(slice(0, 1)),
            "early_nrmse": error(slice(1, 4)),
            "late_nrmse": error(slice(4, 7)),
            "future_nrmse": error(slice(1, 7)),
            "paired_test_episode_nrmse": ep_err.tolist()}


def run(seeds=APPARATUS_SEEDS, epochs=160):
    torch.set_num_threads(1)
    rows = []
    for seed in seeds:
        for world, leak in (("closed", 0.0), ("bridge", .28)):
            data = build_splits(seed, leak)
            for name in MODEL_NAMES:
                item = train_one(data["train"], data["val"], data["test"],
                                 kind=name, apparatus_seed=seed,
                                 leak=leak, epochs=epochs)
                print(f"{seed:3d} {world:6s} {name:12s}: "
                      f"now={item['current_nrmse']:.3f} "
                      f"early={item['early_nrmse']:.3f} "
                      f"late={item['late_nrmse']:.3f} "
                      f"future={item['future_nrmse']:.3f} "
                      f"epochs={item['checkpoint_epoch']}", flush=True)
                rows.append(item)
    return {"protocol": "PROTOCOL_LEARNED.md",
            "apparatus_seeds": list(seeds),
            "architecture": "linear or GRU encoder + tanh-affine action updater and reader",
            "train_count": SPLIT_COUNTS["train"],
            "val_count": SPLIT_COUNTS["val"],
            "test_count": SPLIT_COUNTS["test"],
            "context_frames": HISTORY,
            "sensor_channels": SENSORS,
            "open_loop_future_steps": HORIZON,
            "epochs": epochs, "models": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="results/learned_memory_v2.json")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(APPARATUS_SEEDS))
    args = parser.parse_args()
    report = run(seeds=args.seeds, epochs=args.epochs)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
