"""Optional PyTorch v2 tests: pytest -q tests/test_learned_memory.py."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from learned_memory import (
    make_apparatus, evolve, build_splits, Standardizer, Predictor, HORIZON, QUERIES
)


def test_twelve_sensors_and_disjoint_splits():
    app = make_apparatus(41, .28)
    assert app["sensor"].shape == (12, 32)
    d = build_splits(41, .28)
    assert d["train"].history.shape == (1200, 44)
    assert d["test"].targets.shape == (400, 7, 6)
    assert d["val"].sequence.shape == (240, 3, 16)
    assert not np.array_equal(d["train"].history[0], d["test"].history[0])


def test_identical_present_answers_diverge_after_action():
    app = make_apparatus(41, .28)
    z = np.random.default_rng(73).normal(size=(1, 32))
    twin = z.copy()
    twin[:, 4] += 1.1
    np.testing.assert_allclose(z[:, :4] @ app["reads"].T,
                               twin[:, :4] @ app["reads"].T)
    left = evolve(z, np.array([2]), app)
    right = evolve(twin, np.array([2]), app)
    assert np.max(np.abs(
        left[:, :4] @ app["reads"].T -
        right[:, :4] @ app["reads"].T)) > .03


def test_closed_world_bridge_never_feeds_into_core():
    app = make_apparatus(41, 0.0)
    z = np.random.default_rng(73).normal(size=(1, 32))
    twin = z.copy()
    twin[:, 4] += 1.1
    for a in [3, 0, 2, 1, 3]:
        z = evolve(z, np.array([a]), app)
        twin = evolve(twin, np.array([a]), app)
        np.testing.assert_allclose(z[:, :4], twin[:, :4], atol=1e-12)


def test_all_encoders_use_only_observed_inputs_at_rollout():
    data = build_splits(41, .28)
    x, seq, actions, _ = Standardizer(data["train"]).arrays(data["val"])
    center = x.numpy().mean(axis=0)
    _, _, vt = np.linalg.svd(x.numpy() - center, full_matrices=False)
    for kind in ("predictive4", "predictive5", "pca5", "gru5"):
        model = Predictor(kind, pca=(center, vt[:5]) if kind == "pca5" else None)
        with torch.no_grad():
            out = model(x[:5], seq[:5], actions[:5])
            code = model.encode(x[:5], seq[:5])
        assert out.shape == (5, HORIZON + 1, QUERIES)
        assert code.shape == (5, 4 if kind == "predictive4" else 5)
        assert torch.isfinite(out).all()
