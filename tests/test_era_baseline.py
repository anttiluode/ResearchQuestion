"""ERA baseline tests (optional: learned_memory imports torch; scipy for PEM)."""
import numpy as np
import pytest

pytest.importorskip("torch")

import era_baseline as era  # noqa: E402
from learned_memory import ACTIONS, HORIZON, QUERIES, Episodes  # noqa: E402


def exact_switched_system(n=4, count=4000, seed=0):
    """Linear switched affine system whose past features determine the state."""
    rng = np.random.default_rng(seed)
    trans = [0.8 * np.eye(n) + 0.1 * rng.normal(size=(n, n)) for _ in range(ACTIONS)]
    drive = rng.normal(scale=0.2, size=(ACTIONS, n))
    reader = rng.normal(size=(QUERIES, n))
    mix = rng.normal(size=(n, 44))
    h = rng.normal(size=(count, 44))
    z = h @ mix.T / 7.0
    acts = rng.integers(0, ACTIONS, size=(count, HORIZON))
    ys = [z @ reader.T]
    for t in range(HORIZON):
        z = np.einsum("bij,bj->bi", np.stack(trans)[acts[:, t]], z) + drive[acts[:, t]]
        ys.append(z @ reader.T)
    return h, Episodes(h, None, acts, np.stack(ys, axis=1))


def test_era_recovers_exact_linear_switched_system():
    h, ep = exact_switched_system()
    blocks = era.hankel_blocks(h, ep, depth=2, ridge=1e-9)
    model = era.realize(blocks, depth=2, rank=4)
    pred = era.rollout(model, h, ep.future_actions)
    rel = np.sqrt(np.mean((pred - ep.targets) ** 2) / np.mean(ep.targets ** 2))
    assert rel < 1e-6
    s = model["singular_values"]
    assert s[3] > 1e3 * s[4]          # Hankel rank is exactly the state dimension


def test_era_underfits_below_true_rank():
    h, ep = exact_switched_system()
    blocks = era.hankel_blocks(h, ep, depth=2, ridge=1e-9)
    pred = era.rollout(era.realize(blocks, depth=2, rank=3), h, ep.future_actions)
    rel = np.sqrt(np.mean((pred - ep.targets) ** 2) / np.mean(ep.targets ** 2))
    assert rel > 1e-2


def test_scores_match_v2_metric_definition():
    truth = np.ones((5, 7, QUERIES))
    out = era.scores(truth * 1.1, truth)
    assert abs(out["future_nrmse"] - 0.1) < 1e-12
    assert len(out["paired_test_episode_nrmse"]) == 5
