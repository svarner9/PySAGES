"""The mlmd backend's in-step rebuild thread.

A calculator that rebuilds a data structure from the current coordinates inside every step (the
MACE-FMM partition) exposes ``pysages_in_step_refresh``. The backend then carries that calculator's
extra state through the compiled step and reruns a chunk whose rebuild overflowed its capacities.

The toy here is the protocol, not the FMM: a "partition" that is one scalar recomputed from the
frame, and a capacity that a coordinate can outgrow. Testing against the FMM itself would need a GPU
and a trained model; testing the protocol needs neither, and the threading is what can break.
"""
import numpy as np
import pytest

mlmd = pytest.importorskip("mlmd")

import jax.numpy as jnp  # noqa: E402

import pysages  # noqa: E402
from pysages.backends.core import SamplingContext  # noqa: E402
from pysages.colvars import Component  # noqa: E402
from pysages.methods import HarmonicBias, Unbiased  # noqa: E402

from mlmd import Langevin, System  # noqa: E402
from mlmd.calculators import forces_from_energy  # noqa: E402

mlmd.enable_float64()

_L = 12.0


def _payload(positions):
    """The 'partition': a scalar rebuilt from the frame, exactly as written in both paths."""
    return 1.0 + 1e-3 * jnp.mean(positions)


def _energy(positions, payload):
    return payload * jnp.sum(1.0 - jnp.cos(positions * (2.0 * jnp.pi / _L)))


class ToyRebuildCalculator:
    """Rebuilds ``_payload`` every force call. ``capacity`` is the width the rebuild can hold; a
    coordinate beyond it means the frame did not fit and the chunk has to be rerun wider."""

    r_max = None
    dtype = jnp.float64

    def __init__(self, capacity=np.inf):
        self.capacity = float(capacity)
        self.rebuilds = 0                      # host-side rebuilds, i.e. outside the compiled step

    def build(self, numbers, cell=None, pbc=(False, False, False)):
        return forces_from_energy(lambda R, neighbors: _energy(R, _payload(R)))

    # --- the protocol the backend looks for --------------------------------
    def pysages_in_step_refresh(self, positions):
        self.rebuilds += 1
        return ToyRefresher(self)


class ToyRefresher:
    def __init__(self, calc):
        self.calc = calc
        self.growths = 0

    def fresh(self):
        return (jnp.asarray(self.calc.capacity), jnp.bool_(False), jnp.asarray(0.0))

    def force_fn(self, nl):
        def force(R, cell, extras):
            capacity, overflow, worst = extras
            e, f = forces_from_energy(lambda r, _n: _energy(r, _payload(r)))(R, None)
            need = jnp.max(jnp.abs(R))
            return e, f, (jnp.int32(0), jnp.bool_(False)), (
                capacity, overflow | (need > capacity), jnp.maximum(worst, need))

        return force

    @staticmethod
    def overflowed(extras):
        return bool(extras[1])

    def grow(self, extras):
        self.calc.capacity = float(extras[2]) * 2.0
        self.growths += 1
        return f"capacity -> {self.calc.capacity}"


class FreeRefresher(ToyRefresher):
    """A rebuild that contributes no energy at all, so a trajectory through it is free flight."""

    def force_fn(self, nl):
        def force(R, cell, extras):
            capacity, overflow, worst = extras
            return (jnp.asarray(0.0), jnp.zeros_like(R),
                    (jnp.int32(0), jnp.bool_(False)), (capacity, overflow, worst))

        return force


def toy_dynamics(capacity=np.inf, seed=0, n=8):
    rng = np.random.default_rng(1)
    positions = rng.uniform(-2.0, 2.0, size=(n, 3))
    system = System(positions, [2] * n, ToyRebuildCalculator(capacity), masses=[4.0] * n)
    return Langevin(system, timestep=1.0, temperature=300.0, friction=0.05, seed=seed)


def test_unbiased_with_in_step_rebuild_reproduces_the_plain_mlmd_trajectory():
    """`Unbiased` plus the rebuild thread must reproduce the plain-mlmd trajectory: the carried
    rebuild changes what the step *carries*, never what it computes.

    Not bit for bit: threading the rebuild's report through the carry puts more operations in the
    same XLA computation, which refuses it a fusion the plain step gets, and the two then round a
    reduction differently. The observed gap after 200 steps is ~1 ulp, so the tolerance below is
    roundoff and the test would still catch a stale or missing rebuild by many orders of magnitude
    (`test_the_compiled_step_calls_the_refreshers_force` pins that directly)."""
    reference = toy_dynamics(seed=3)
    reference.run(200)

    dyn = toy_dynamics(seed=3)
    context = SamplingContext(Unbiased([Component([0], 0)]), lambda: dyn)
    assert context.sampler.refresher is not None            # auto-detected from the calculator
    context.run(200, stride=50)

    np.testing.assert_allclose(np.asarray(dyn.state.positions),
                               np.asarray(reference.state.positions), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(dyn.state.velocities),
                               np.asarray(reference.state.velocities), rtol=1e-12, atol=1e-14)
    assert dyn.nsteps == reference.nsteps == 200


def test_the_compiled_step_calls_the_refreshers_force():
    """The step must take its energy and forces from the refresher, not from the calculator's own
    `build()`. Here they disagree on purpose: `build()` is a stiff spring, the refresher is free
    flight. With the thread on, the particles must leave the well."""
    class Contradictory(ToyRebuildCalculator):
        def build(self, numbers, cell=None, pbc=(False, False, False)):
            return forces_from_energy(lambda R, neighbors: 50.0 * jnp.sum(R ** 2))

        def pysages_in_step_refresh(self, positions):
            self.rebuilds += 1
            return FreeRefresher(self)

    def context_for(in_step):
        rng = np.random.default_rng(1)
        positions = rng.uniform(-0.05, 0.05, size=(64, 3))
        system = System(positions, [2] * 64, Contradictory(), masses=[4.0] * 64)
        dyn = Langevin(system, timestep=1.0, temperature=300.0, friction=0.05, seed=11)
        return dyn, SamplingContext(Unbiased([Component([0], 0)]), lambda: dyn,
                                    in_step_refresh=in_step)

    free_dyn, free_ctx = context_for(True)
    free_ctx.run(400, stride=400)
    bound_dyn, bound_ctx = context_for(False)
    bound_ctx.run(400, stride=400)

    free_spread = float(np.sqrt(np.mean(np.asarray(free_dyn.state.positions) ** 2)))
    bound_spread = float(np.sqrt(np.mean(np.asarray(bound_dyn.state.positions) ** 2)))
    assert bound_spread < 0.05                      # kT / (2 * 50) is a very tight well
    assert free_spread > 20 * bound_spread          # free flight for 400 fs


def test_capacity_overflow_reruns_the_chunk_and_recovers_the_same_trajectory():
    """A chunk whose rebuild overflowed ran on a truncated structure, so it is discarded: the
    capacity grows and the chunk reruns from its own start carry. Same seed, so the recovered
    trajectory is the one the never-overflowing run produced."""
    reference = toy_dynamics(seed=5)
    reference.run(200)

    dyn = toy_dynamics(capacity=0.5, seed=5)                # every frame overflows at first
    context = SamplingContext(Unbiased([Component([0], 0)]), lambda: dyn)
    context.run(200, stride=50, frames_per_chunk=2)

    assert context.sampler.refresher.growths >= 1
    assert dyn.system.calculator.capacity > 0.5
    np.testing.assert_allclose(np.asarray(dyn.state.positions),
                               np.asarray(reference.state.positions), rtol=1e-12, atol=1e-14)
    assert dyn.nsteps == 200


def test_bias_still_acts_with_full_weight_through_the_rebuild_thread():
    """The rebuild thread must not disturb the bias bookkeeping: a harmonic bias on x of a free
    particle gives <x^2> = kT / k_bias, not kT / (k_bias / 2)."""
    from mlmd import units

    k_bias, T = 6.0, 300.0
    kT = units.kB * T
    n = 200
    rng = np.random.default_rng(2)
    positions = np.zeros((n, 3))

    class Free(ToyRebuildCalculator):
        def build(self, numbers, cell=None, pbc=(False, False, False)):
            return forces_from_energy(lambda R, neighbors: 0.0 * jnp.sum(R))

    calc = Free()
    system = System(positions, [2] * n, calc, masses=[4.0] * n)
    dyn = Langevin(system, timestep=1.0, temperature=T, friction=0.05, seed=7)

    calc.pysages_in_step_refresh = lambda positions: FreeRefresher(calc)

    cvs = [Component([i], 0) for i in range(n)]
    method = HarmonicBias(cvs, kspring=k_bias, center=np.zeros(n))
    xi = []
    context = SamplingContext(method, lambda: dyn,
                              lambda snapshot, state, step: xi.append(np.asarray(state.xi).ravel()))
    context.run(30000, stride=50)

    x = np.array(xi[len(xi) // 5:])
    variance = float(np.mean(x ** 2))
    assert abs(variance / (kT / k_bias) - 1) < 0.05
    assert abs(variance / (kT / (0.5 * k_bias)) - 1) > 0.3      # rules out half weight


def test_in_step_refresh_can_be_switched_off_and_is_not_a_run_argument():
    dyn = toy_dynamics(seed=3)
    context = SamplingContext(Unbiased([Component([0], 0)]), lambda: dyn, in_step_refresh=False)
    assert context.sampler.refresher is None
    assert dyn.system.calculator.rebuilds == 0
    context.run(20, stride=20)

    other = toy_dynamics(seed=3)
    context = SamplingContext(Unbiased([Component([0], 0)]), lambda: other)
    with pytest.raises(TypeError, match="SamplingContext argument"):
        context.run(20, stride=20, in_step_refresh=False)
