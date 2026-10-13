import numpy as np
import pytest

from research_question import (make_world, random_full, local_future_lens,
                               lens_dimension, projection_distance,
                               full_trajectory, query_trajectory)


@pytest.mark.parametrize("leak", [0.0, 0.28])
@pytest.mark.parametrize("seed", [13, 31])
def test_sieve_telescopes_to_exact_input(seed, leak):
    w = make_world(seed=seed, leak=leak)
    x = random_full(w, np.random.default_rng(555))
    core, bridge, noise = w.sieve(x)
    np.testing.assert_allclose(x, core+bridge+noise, rtol=1e-12, atol=1e-12)


def test_inert_residue_never_changes_any_answer():
    w = make_world(leak=0.28)
    rng = np.random.default_rng(3)
    x = random_full(w, rng)
    other = x + w.q[:, w.core_dim + 1:] @ rng.normal(size=w.dimension-w.core_dim-1)
    for actions in [[0, 1, 2, 3, 2], [3, 3, 0, 0]]:
        np.testing.assert_allclose(full_trajectory(w, x, actions),
                                   full_trajectory(w, other, actions), atol=2e-14)


@pytest.mark.parametrize("leak,keep_bridge", [(0.0, False), (0.0, True), (0.28, True)])
def test_future_code_is_closed_when_sufficient(leak, keep_bridge):
    w = make_world(leak=leak)
    x = random_full(w, np.random.default_rng(999))
    actions = [0, 3, 2, 1] * 4
    np.testing.assert_allclose(query_trajectory(w, x, actions, keep_bridge),
                               full_trajectory(w, x, actions), atol=2e-14)


def test_current_answer_code_is_not_future_sufficient():
    w = make_world(leak=0.28)
    x = random_full(w, np.random.default_rng(123))
    twin = x + 1.3*w.q[:, w.core_dim]
    np.testing.assert_allclose([w.read(x, q) for q in range(w.query_count)],
                               [w.read(twin, q) for q in range(w.query_count)], atol=1e-14)
    response_diffs = [abs(w.read(w.step(x, a), q) -
                          w.read(w.step(twin, a), q))
                      for a in range(w.action_count) for q in range(w.query_count)]
    assert max(response_diffs) > 0.02


@pytest.mark.parametrize("leak,expected", [(0.0, 4), (0.28, 5)])
def test_one_step_jacobian_lens_finds_future_relevant_subspace(leak, expected):
    w = make_world(seed=13, leak=leak)
    ctx = np.stack([random_full(w, np.random.default_rng(i)) for i in [11, 12, 13]])
    singular, vt = local_future_lens(w, ctx)
    assert lens_dimension(singular) == expected
    target = w.q[:, :expected].T
    assert projection_distance(vt[:expected], target) < 1e-6
