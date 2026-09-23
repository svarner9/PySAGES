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

Calculators that rebuild state per step
---------------------------------------
A calculator may need to rebuild a data structure from the current coordinates inside every step -- the
MACE-FMM calculator rebuilds its fast-multipole partition that way, so the long-range interaction is
never evaluated on a partition built for an older frame. Such a calculator exposes
``pysages_in_step_refresh(positions)``; the runner then threads that calculator's own extra carry
element through the compiled step alongside the sampling state, and reruns a chunk whose carry reports
that the rebuild overflowed its capacities (the chunk ran on a truncated structure, so it is not a
trajectory). Pass ``in_step_refresh=False`` to ``SamplingContext`` to hold the structure instead.

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
        """Resume: put the saved positions/velocities/forces back into the dynamics, plus the
        step counter and PRNG base key (carried in ``prev_snapshot.extras``, see
        ``take_snapshot``/``update_snapshot`` below).

        Without this, a resumed run always starts its underlying ``mlmd`` ``State.step`` back
        at 0: the step's PRNG key is ``fold_in(base_key, step)`` (mlmd/runner.py), so the
        resumed segment silently *repeats* the first segment's noise sequence instead of
        continuing it, and the owed bias half-kick in ``build_runner.step`` (gated on
        ``state.step > 0``) is skipped on the very first resumed step even though a bias was
        in effect when the run was checkpointed. Restoring ``step``/``key`` here makes both
        exact: `fold_in` reproduces the same key it would have on an uninterrupted run, and the
        half-kick fires immediately, exactly as T2's design memo (Sec. 5, "Backend blocker for
        restart bit-exact") requires for a bit-exact resumed, biased trajectory on CPU.
        """
        self.snapshot = prev_snapshot
        velocities, _ = prev_snapshot.vel_mass
        state = self.dynamics.state
        replacements = dict(
            positions=np.asarray(prev_snapshot.positions, state.positions.dtype),
            velocities=np.asarray(velocities, state.velocities.dtype),
            forces=np.asarray(prev_snapshot.forces, state.forces.dtype),
        )
        extras = prev_snapshot.extras or {}
        if "step" in extras:
            replacements["step"] = np.asarray(extras["step"], state.step.dtype)
        if "key" in extras:
            replacements["key"] = np.asarray(extras["key"], state.key.dtype)
        self.dynamics.state = state._replace(**replacements)

    def take_snapshot(self):
        return copy(self.snapshot)


def _pysages_dtype():
    # PySAGES enables x64 on import and initializes its states in float64; the MD state may be float32.
    # Handing it a float64 snapshot keeps every method-state leaf at one dtype across steps.
    return np.float64 if jax.config.jax_enable_x64 else np.float32


def take_snapshot(dynamics):
    """Snapshot of the dynamics' current device state (velocities, not momenta, in ``vel_mass``).

    ``extras`` carries the mlmd ``State.step``/``.key`` alongside the usual
    positions/velocities/forces, purely so a later ``Sampler.restore`` (resume) can put them
    back -- see the docstring there. Nothing else reads ``extras`` on this backend.
    """
    dtype = _pysages_dtype()
    state = dynamics.state
    system = dynamics.system
    masses = dynamics.masses.reshape(-1, 1).astype(dtype)
    ids = np.arange(system.n_atoms)
    cell = np.eye(3) if system.cell is None else np.asarray(system.cell)
    box = Box(cell, (0.0, 0.0, 0.0))
    extras = {"step": state.step, "key": state.key}
    return Snapshot(state.positions.astype(dtype), (state.velocities.astype(dtype), masses),
                    state.forces.astype(dtype), ids, box, dynamics.dt, extras)


def update_snapshot(snapshot, state, velocities):
    dtype = _pysages_dtype()
    _, masses = snapshot.vel_mass
    extras = {"step": state.step, "key": state.key}
    return snapshot._replace(positions=state.positions.astype(dtype),
                             vel_mass=(velocities.astype(dtype), masses),
                             forces=state.forces.astype(dtype),
                             extras=extras)


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


def build_runner(dynamics, sampler, refresher=None):
    from mlmd.runner import advance_state, frames_snapshot

    dt = dynamics.dt
    masses = dynamics.masses
    m = masses[:, None]

    def make_step():
        # Both are rebuilt after a neighbor-list regrowth, which is why they are read here:
        # ChunkRunner calls make_step again then.
        integrator_step = dynamics.integrator_step if refresher is None else None
        force_fn = None if refresher is None else refresher.force_fn(dynamics._nl)

        def step(carry):
            state, snapshot, method_state = carry[:3]
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
            if refresher is None:
                state = advance_state(state, integrator_step, velocities=velocities, forces=forces)
                return state, snapshot, method_state
            # The calculator rebuilds from the coordinates the step hands it and reports what its
            # capacities had to hold; `holder` lifts that report back out of the force call.
            extras = carry[3]
            holder = []

            def force(R, cell):
                e, f, aux, extras_out = force_fn(R, cell, extras)
                holder.append(extras_out)
                return e, f, aux

            state = advance_state(state, dynamics.make_step(force),
                                  velocities=velocities, forces=forces)
            return state, snapshot, method_state, holder[-1]

        return step

    def snapshot_fn(carry):
        state, snapshot, method_state = carry[:3]
        return frames_snapshot(state, masses), snapshot, method_state

    runner = dynamics.make_runner(make_step, snapshot_fn)

    def run(timesteps, stride=None, frames_per_chunk=10, **kwargs):
        nonlocal runner
        if "in_step_refresh" in kwargs:
            # It selects how the step is compiled, so it is decided at bind time; accepting it here
            # would silently ignore it.
            raise TypeError("in_step_refresh is a SamplingContext argument, not a run() argument: "
                            "SamplingContext(method, context, in_step_refresh=False)")
        timesteps = int(timesteps)
        if stride is None:
            intervals = [interval for _, interval in dynamics.observers]
            stride = math.gcd(*intervals) if intervals else 100
        stride = max(1, min(int(stride), timesteps))
        carry = (dynamics.state, sampler.snapshot, sampler.state)
        target = dynamics.nsteps + timesteps
        while dynamics.nsteps < target:
            remaining = target - dynamics.nsteps
            # Land on the next ABSOLUTE multiple of the stride, by shortening the first chunk if
            # the run does not start on one. mlmd fires an observer only when `frame.step % interval`
            # is zero, so frames offset from the boundary -- which is what a preceding odd number of
            # steps leaves, e.g. a one-step pre-flight -- match no interval at all and the observers
            # stay silent for the whole run.
            to_boundary = stride - dynamics.nsteps % stride
            if to_boundary != stride or remaining < stride:
                this_stride, n_frames = min(remaining, to_boundary), 1
            else:
                n_frames = min(frames_per_chunk, remaining // stride)
                this_stride = stride
            if refresher is None:
                carry, (frames, snapshots, states) = runner.advance(
                    carry, this_stride, n_frames)
            else:
                # `fresh()` is re-read on every attempt: a growth moves the capacities it reports.
                start = carry[:3] + (refresher.fresh(),)
                while True:
                    carry, (frames, snapshots, states) = runner.advance(
                        start, this_stride, n_frames)
                    if not refresher.overflowed(carry[3]):
                        break
                    refresher.grow(carry[3])
                    runner = dynamics.make_runner(make_step, snapshot_fn)
                    run.runner = runner
                    start = start[:3] + (refresher.fresh(),)
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
    refresher = None
    if kwargs.get("in_step_refresh", True):
        factory = getattr(dynamics.system.calculator, "pysages_in_step_refresh", None)
        if factory is not None:
            refresher = factory(dynamics.state.positions)
    snapshot = take_snapshot(dynamics)
    helpers = build_helpers(dynamics, sampling_method)
    method_bundle = sampling_method.build(snapshot, helpers)
    sampler = Sampler(method_bundle, dynamics, callback)
    sampling_context.run = build_runner(dynamics, sampler, refresher)
    sampler.refresher = refresher
    return sampler
