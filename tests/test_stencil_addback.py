# SPDX-License-Identifier: MIT
"""
Three-step bias add-back (``stencil_addback``) for ABF / MetaDSpectralABF on the mlmd backend.

The mlmd backend reports p_n = m v_n + dt/2 b_{n-1} and applies each bias b_k as a full kick at
x_k, so the bias impulse inside the second-order backward difference
``(1.5 Wp_n - 2 Wp_{n-1} + 0.5 Wp_{n-2}) / dt`` is ``-(2.25 f_{n-1} - 1.5 f_{n-2} + 0.25 f_{n-3})``.
Adding back only ``f_{n-1}`` (legacy) is exact for a smooth bias but not where the applied CV force
jumps -- e.g. at the grid edge, where the bias switches to the window restraint.

Deterministic check: a free particle (U = 0, friction 0 -> exact velocity Verlet) bouncing off the
out-of-grid restraint. The physical mean force is exactly zero, so with the stencil every binned
sample is zero to round-off (after the first three updates, whose backward difference starts from
zeroed history); with the legacy add-back the bins next to the grid edges are not.
"""

import numpy as onp
import pytest

mlmd = pytest.importorskip("mlmd")

import jax.numpy as jnp  # noqa: E402
import pysages  # noqa: E402
from pysages.colvars import Component  # noqa: E402
from pysages.grids import build_indexer  # noqa: E402
from pysages.methods import ABF, CVRestraints, MetaDSpectralABF  # noqa: E402

from test_metad_sabf import _one_particle  # noqa: E402

STEPS = 6000
GRID = dict(lower=(-0.5,), upper=(0.5,), shape=(20,))
RESTRAINTS = CVRestraints(lower=(-0.3,), upper=(0.3,), kl=(20.0,), ku=(20.0,))  # jump 4 eV/A


def _free(_positions):
    return jnp.zeros(())


def _ctx():
    dyn = _one_particle(_free, mass=1.0, dt=0.5, friction=0.0, seed=7)
    # Deterministic start: friction 0 leaves velocities at their initial value, so give the
    # particle a speed along x (ASE units; ~0.0025 A/step, many wall bounces in STEPS).
    dyn.system.set_velocities(onp.array([[0.05, 0.0, 0.0]]))
    return dyn


def _metad(stencil):
    grid = pysages.Grid(**GRID, periodic=False)
    return MetaDSpectralABF(
        [Component([0], 0)], grid, height=0.0, sigma=0.1, stride=10**9, ngaussians=2,
        deltaT=None, abf_bias=False, burn_in=3, N=10**9, fit_threshold=10**9,
        restraints=RESTRAINTS, stencil_addback=stencil,
    )


def _abf(stencil):
    grid = pysages.Grid(**GRID, periodic=False)
    return ABF([Component([0], 0)], grid, N=10**9, restraints=RESTRAINTS, stencil_addback=stencil)


@pytest.mark.parametrize("stencil", [True, False])
def test_metad_sabf_stencil_addback_free_particle(stencil):
    result = pysages.run(_metad(stencil), _ctx, STEPS)
    state = result.states[0]
    hist = onp.asarray(state.hist_prod).reshape(-1)
    fsum = onp.asarray(state.Fsum_prod).reshape(-1)
    assert hist[0] > 0 and hist[-1] > 0, "particle never reached the grid edges"
    mean = onp.where(hist > 0, fsum / onp.maximum(hist, 1), 0.0)
    if stencil:
        assert onp.max(onp.abs(mean)) < 1e-8, mean
    else:
        assert onp.max(onp.abs(mean[[0, -1]])) > 1e-2, mean  # legacy: edge bins contaminated


@pytest.mark.parametrize("stencil", [True, False])
def test_abf_stencil_addback_free_particle(stencil):
    method = _abf(stencil)
    result = pysages.run(method, _ctx, STEPS)
    state = result.states[0]
    hist = onp.asarray(state.hist).reshape(-1)
    fsum = onp.asarray(state.Fsum).reshape(-1)
    assert hist[0] > 0 and hist[-1] > 0, "particle never reached the grid edges"
    # The first three samples (zeroed Wp history) all sit in the bin containing the start, x = 0.
    start = int(onp.asarray(build_indexer(method.grid)(jnp.array([0.0])))[0])
    keep = onp.ones_like(hist, dtype=bool)
    keep[max(start - 1, 0):start + 2] = False
    mean = onp.where(hist > 0, fsum / onp.maximum(hist, 1), 0.0)
    if stencil:
        assert onp.max(onp.abs(mean[keep])) < 1e-8, mean
    else:
        assert onp.max(onp.abs(mean[[0, -1]])) > 1e-2, mean
