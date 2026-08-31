# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
PySAGES backend for mlmd, the JAX machine-learning-potential MD engine.

The context is an ``mlmd.Dynamics`` object (``Langevin`` or ``VelocityVerlet``) built exactly as for an
unbiased run; ``generate_context`` just returns it. The whole biased step — neighbor list, potential,
collective variables, sampling-method update, bias, integrator — is one compiled JAX program, run in
chunks of many steps per device call (mlmd's fused ``lax.scan``), so enhanced sampling costs about as
much as plain MD. Observers attached to the dynamics (``Trajectory``, ``Logger``, ...) keep working.

How the bias enters
-------------------
BAOAB (and velocity Verlet) apply each force twice: half a kick at the end of one step and half a kick
at the start of the next. The sampling method evaluates the bias from the positions at the start of a
step, so the previous step's closing half-kick could not include it. The runner therefore adds that
missing half-kick before the step and hands the integrator ``forces + bias`` for its opening half-kick:
the bias acts with full weight ``dt``, evaluated at one set of positions, exactly as if it were part of
the force field. (Adding the bias *after* the step, as a naive adaptation of the jax-md backend would,
gives it half weight and makes ABF's mean-force estimate converge to twice the true value.)

Callbacks
---------
The Python ``callback(snapshot, state, timestep)`` is called once per *frame* — every ``stride``
steps — with the sampling state and snapshot of that frame. ``stride`` defaults to the greatest common
divisor of the intervals of the observers attached to the dynamics (100 if there are none); pass
``pysages.run(..., stride=..., frames_per_chunk=...)`` to change it.
"""

import math

import jax
from jax import numpy as np

from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.typing import Callable
from pysages.utils import copy


class Sampler:
    def __init__(self, method_bundle, dynamics, callback: Callable):
        initial_snapshot, initialize, method_update = method_bundle
        self.state = initialize()
        self.callback = callback
        self.dynamics = dynamics
        self.snapshot = initial_snapshot
        self.update = method_update

    def restore(self, prev_snapshot):
        """Resume: put the saved positions/velocities/forces back into the dynamics."""
        self.snapshot = prev_snapshot
        velocities, _ = prev_snapshot.vel_mass
        state = self.dynamics.state
        self.dynamics.state = state._replace(
            positions=np.asarray(prev_snapshot.positions, state.positions.dtype),
            velocities=np.asarray(velocities, state.velocities.dtype),
            forces=np.asarray(prev_snapshot.forces, state.forces.dtype),
        )

    def take_snapshot(self):
        return copy(self.snapshot)


def _pysages_dtype():
    # PySAGES enables x64 on import and initializes its states in float64; the MD state may be float32.
    # Handing it a float64 snapshot keeps every method-state leaf at one dtype across steps.
    return np.float64 if jax.config.jax_enable_x64 else np.float32


def take_snapshot(dynamics):
    """Snapshot of the dynamics' current device state (velocities, not momenta, in ``vel_mass``)."""
    dtype = _pysages_dtype()
    state = dynamics.state
    system = dynamics.system
    masses = dynamics.masses.reshape(-1, 1).astype(dtype)
    ids = np.arange(system.n_atoms)
    cell = np.eye(3) if system.cell is None else np.asarray(system.cell)
    box = Box(cell, (0.0, 0.0, 0.0))
    return Snapshot(state.positions.astype(dtype), (state.velocities.astype(dtype), masses),
                    state.forces.astype(dtype), ids, box, dynamics.dt)


def update_snapshot(snapshot, state, velocities):
    dtype = _pysages_dtype()
    _, masses = snapshot.vel_mass
    return snapshot._replace(positions=state.positions.astype(dtype),
                             vel_mass=(velocities.astype(dtype), masses),
                             forces=state.forces.astype(dtype))


def build_snapshot_methods(context, sampling_method):
    def indices(snapshot):
        return snapshot.ids

    def masses(snapshot):
        _, M = snapshot.vel_mass
        return M

    def positions(snapshot):
        return snapshot.positions

    def momenta(snapshot):
        V, M = snapshot.vel_mass
        return (V * M).flatten()

    return SnapshotMethods(positions, indices, jax.jit(momenta), masses)


def build_helpers(context, sampling_method):
    def dimensionality():
        return 3

    snapshot_methods = build_snapshot_methods(context, sampling_method)
    flags = sampling_method.snapshot_flags
    return HelperMethods(build_data_querier(snapshot_methods, flags), dimensionality)


def build_runner(dynamics, sampler):
    from mlmd.runner import advance_state, frames_snapshot

    dt = dynamics.dt
    masses = dynamics.masses
    m = masses[:, None]

    def make_step():
        integrator_step = dynamics.integrator_step   # rebuilt after a neighbor-list regrowth

        def step(carry):
            state, snapshot, method_state = carry
            dtype = state.forces.dtype           # PySAGES works in float64; the MD state may be float32
            prev_bias = method_state.bias
            # The previous closing half-kick used the bare force, so half a kick of bias is owed.
            # Report velocities to the method with the previous bias' half-kick added (a second-order
            # estimate of the true BAOAB velocities), then pay the owed half-kick with the fresh bias.
            velocities = state.velocities
            if prev_bias is not None:
                half_kick = (0.5 * dt * prev_bias / m).astype(dtype)
                velocities = velocities + np.where(state.step > 0, half_kick, 0.0).astype(dtype)
            snapshot = update_snapshot(snapshot, state, velocities)
            method_state = sampler.update(snapshot, method_state)
            bias = method_state.bias
            forces = state.forces
            if bias is not None:
                half_kick = (0.5 * dt * bias / m).astype(dtype)
                velocities = state.velocities + np.where(state.step > 0, half_kick, 0.0).astype(dtype)
                forces = forces + bias.astype(dtype)
            state = advance_state(state, integrator_step, velocities=velocities, forces=forces)
            return state, snapshot, method_state

        return step

    def snapshot_fn(carry):
        state, snapshot, method_state = carry
        return frames_snapshot(state, masses), snapshot, method_state

    runner = dynamics.make_runner(make_step, snapshot_fn)

    def run(timesteps, stride=None, frames_per_chunk=10, **kwargs):
        timesteps = int(timesteps)
        if stride is None:
            intervals = [interval for _, interval in dynamics.observers]
            stride = math.gcd(*intervals) if intervals else 100
        stride = max(1, min(int(stride), timesteps))
        carry = (dynamics.state, sampler.snapshot, sampler.state)
        target = dynamics.nsteps + timesteps
        while dynamics.nsteps < target:
            remaining = target - dynamics.nsteps
            n_frames = min(frames_per_chunk, remaining // stride) if remaining >= stride else 0
            this_stride = stride if n_frames > 0 else remaining
            carry, (frames, snapshots, states) = runner.advance(carry, this_stride, max(n_frames, 1))
            dynamics.state = carry[0]
            sampler.snapshot, sampler.state = carry[1], carry[2]
            for k, frame in enumerate(dynamics.frames_to_host(frames)):
                dynamics.fire_observers(frame)
                if sampler.callback:
                    frame_snapshot = jax.tree_util.tree_map(lambda x: x[k], snapshots)
                    frame_state = jax.tree_util.tree_map(lambda x: x[k], states)
                    sampler.callback(frame_snapshot, frame_state, frame.step)
        # the carried snapshot is the one the method saw *before* the last step; the Result (and a
        # later resume) must describe the final state
        state = dynamics.state
        sampler.snapshot = update_snapshot(sampler.snapshot, state, state.velocities)

    run.runner = runner          # exposed for tests and diagnostics
    return run


def bind(sampling_context: SamplingContext, callback: Callable, **kwargs):
    dynamics = sampling_context.context
    sampling_method = sampling_context.method
    from mlmd import NPT

    if isinstance(dynamics, NPT):
        raise ValueError(
            "PySAGES + mlmd NPT is not supported: the bias force carries no virial, so the "
            "barostat would see the biased forces but an unbiased stress and sample a subtly "
            "wrong ensemble. Run the biased sampling at constant volume (Langevin), or ask for "
            "biased-NPT support explicitly."
        )
    dynamics.prepare()
    if dynamics.nsteps == 0:                     # like mlmd's own run(): observers see the initial frame
        dynamics.fire_observers(dynamics.current_frame())
    snapshot = take_snapshot(dynamics)
    helpers = build_helpers(dynamics, sampling_method)
    method_bundle = sampling_method.build(snapshot, helpers)
    sampler = Sampler(method_bundle, dynamics, callback)
    sampling_context.run = build_runner(dynamics, sampler)
    return sampler
