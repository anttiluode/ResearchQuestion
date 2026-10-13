"""A constructed nonlinear dynamical system with future-sufficient quotients.

Hidden coordinates: 4 answer-relevant values, one bridge, 27 nuisance axes.
This is a mechanism isolator, not an automatically learned neural memory.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class World:
    q: np.ndarray
    a: np.ndarray
    coupling: np.ndarray
    reads: np.ndarray
    rho: np.ndarray
    h: np.ndarray
    nuisance_drive: np.ndarray
    leak: float

    @property
    def dimension(self) -> int:
        return self.q.shape[0]

    @property
    def core_dim(self) -> int:
        return self.a.shape[1]

    @property
    def action_count(self) -> int:
        return len(self.a)

    @property
    def query_count(self) -> int:
        return len(self.reads)

    def hidden(self, full: np.ndarray) -> np.ndarray:
        return self.q.T @ full

    def full(self, hidden: np.ndarray) -> np.ndarray:
        return self.q @ hidden

    def read(self, full: np.ndarray, query: int) -> float:
        return self.read_code(self.hidden(full)[:self.core_dim], query)

    def read_code(self, code: np.ndarray, query: int) -> float:
        return float(self.reads[query] @ code[:self.core_dim])

    def code(self, full: np.ndarray, retain_bridge: bool) -> np.ndarray:
        return self.hidden(full)[:self.core_dim + int(retain_bridge)].copy()

    def step_code(self, code: np.ndarray, action: int) -> np.ndarray:
        """Update the compact code directly, without reopening the full state.

        Missing bridge is zero-filled as an intentionally incorrect control.
        """
        k = self.core_dim
        s = code[:k]
        b = float(code[k]) if len(code) > k else 0.0
        next_s = np.tanh(self.a[action] @ s + 0.13 * np.sin(s)
                         + self.leak * self.coupling[action] * b)
        if len(code) == k:
            return next_s
        next_b = self.rho[action] * b + 0.11 * np.tanh(s[0])
        return np.concatenate([next_s, [next_b]])

    def step(self, full: np.ndarray, action: int) -> np.ndarray:
        z = self.hidden(full)
        k = self.core_dim
        s, b, nuisance = z[:k], float(z[k]), z[k + 1:]
        code = self.step_code(np.concatenate([s, [b]]), action)
        next_nuisance = np.tanh(self.h[action] @ nuisance
                                + 0.06 * self.nuisance_drive[action] * b)
        return self.full(np.concatenate([code, next_nuisance]))

    def sieve(self, full: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Two nested exact projection steps: full -> (s,b) -> s.

        This is a telescoping residue ledger, NOT Sihti's dyadic diffusion.
        """
        z = self.hidden(full)
        k = self.core_dim
        core = self.q[:, :k] @ z[:k]
        bridge = self.q[:, k] * z[k]
        nuisance = full - core - bridge
        return core, bridge, nuisance


def make_world(seed: int = 20261013, dimension: int = 32,
               core_dim: int = 4, leak: float = 0.0,
               action_count: int = 4, query_count: int = 6) -> World:
    if not (2 <= core_dim <= dimension - 2):
        raise ValueError("Require core, one bridge, and at least one nuisance axis")
    if action_count < 2 or query_count < core_dim:
        raise ValueError("Require at least two actions and enough queries")
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    q = q * np.sign(np.diag(r))
    a = np.stack([0.42 * np.eye(core_dim) + 0.15 * rng.normal(
        size=(core_dim, core_dim)) / np.sqrt(core_dim)
        for _ in range(action_count)])
    coupling = rng.normal(size=(action_count, core_dim))
    coupling /= np.linalg.norm(coupling, axis=1, keepdims=True)
    reads = rng.normal(size=(query_count, core_dim))
    reads /= np.linalg.norm(reads, axis=1, keepdims=True)
    rho = rng.uniform(0.55, 0.9, size=action_count)
    d = dimension - core_dim - 1
    h = np.stack([0.43 * np.eye(d) + 0.08 * rng.normal(size=(d, d)) / np.sqrt(d)
                  for _ in range(action_count)])
    nuisance_drive = rng.normal(size=(action_count, d))
    return World(q, a, coupling, reads, rho, h, nuisance_drive, leak)


def random_full(world: World, rng: np.random.Generator) -> np.ndarray:
    k = world.core_dim
    hidden = np.concatenate([rng.normal(scale=0.65, size=k),
                             rng.normal(scale=0.8, size=1),
                             rng.normal(scale=0.8, size=world.dimension - k - 1)])
    return world.full(hidden)


def physical_step_jacobian(world: World, full: np.ndarray, action: int,
                           delta: float = 1e-6) -> np.ndarray:
    """Full ambient dynamical Jacobian; diagnostic, not cheap readout."""
    basis = np.eye(world.dimension)
    return np.stack([(world.step(full + delta*basis[i], action) -
                      world.step(full - delta*basis[i], action)) / (2*delta)
                     for i in range(world.dimension)], axis=1)


def local_future_lens(world: World, contexts: np.ndarray,
                      delta: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Privileged per-coordinate probes of answers after one action.

    The lens requires perturbing every physical coordinate and does not
    demonstrate that a compact encoder can be learned with limited sensors.
    """
    n = world.dimension
    rows = []
    basis = np.eye(n)
    for x in contexts:
        for action in range(world.action_count):
            for i in range(n):
                plus = world.step(x + delta * basis[i], action)
                minus = world.step(x - delta * basis[i], action)
                if i == 0:
                    jac = np.zeros((world.query_count, n))
                jac[:, i] = np.array([world.read(plus, q) - world.read(minus, q)
                                      for q in range(world.query_count)]) / (2*delta)
            rows.extend(jac)
    _, singular, vt = np.linalg.svd(np.asarray(rows), full_matrices=False)
    return singular, vt


def query_trajectory(world: World, full: np.ndarray, actions: list[int],
                     retain_bridge: bool) -> np.ndarray:
    code = world.code(full, retain_bridge)
    results = []
    for action in actions:
        results.extend(world.read_code(code, q) for q in range(world.query_count))
        code = world.step_code(code, action)
    results.extend(world.read_code(code, q) for q in range(world.query_count))
    return np.asarray(results)


def full_trajectory(world: World, full: np.ndarray, actions: list[int]) -> np.ndarray:
    state = full.copy()
    results = []
    for action in actions:
        results.extend(world.read(state, q) for q in range(world.query_count))
        state = world.step(state, action)
    results.extend(world.read(state, q) for q in range(world.query_count))
    return np.asarray(results)


def normed_rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction-truth)**2)) /
                 max(np.sqrt(np.mean(truth**2)), 1e-12))


def lens_dimension(singular: np.ndarray, relative_threshold: float = 1e-6) -> int:
    return int(np.sum(singular > singular[0] * relative_threshold))


def projection_distance(learned: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(learned.T @ learned - truth.T @ truth, ord=2))
