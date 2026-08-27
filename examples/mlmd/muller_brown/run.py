#!/usr/bin/env python3
"""The complete pattern in one screen: SpectralABF with the mlmd backend on an analytic surface.

One particle on the 2-D Müller–Brown potential, written as a ten-line mlmd calculator; the collective
variables are its x and y. No MLIP, no GPU needed — runs in about a minute on a CPU.
"""
import jax.numpy as jnp
import numpy as np

import pysages
from pysages.colvars import Component
from pysages.methods import SpectralABF

from mlmd import Langevin, System
from mlmd.calculators import forces_from_energy

TIMESTEPS = 200_000

# Müller–Brown parameters, scaled so the barriers are ~8 kT at 300 K
A = jnp.array([-200.0, -100.0, -170.0, 15.0])
a = jnp.array([-1.0, -1.0, -6.5, 0.7])
b = jnp.array([0.0, 0.0, 11.0, 0.6])
c = jnp.array([-10.0, -10.0, -6.5, 0.7])
x0 = jnp.array([1.0, 0.0, -0.5, -1.0])
y0 = jnp.array([0.0, 0.5, 1.5, 1.0])
SCALE = 0.002


class MullerBrown:
    """An mlmd calculator. `r_max=None`: no neighbor list; `build` returns ef(positions, neighbors)."""

    r_max = None
    dtype = jnp.float32

    def build(self, numbers, cell=None, pbc=(False, False, False)):
        def energy(positions, neighbors):
            dx, dy = positions[0, 0] - x0, positions[0, 1] - y0
            return SCALE * jnp.sum(A * jnp.exp(a * dx * dx + b * dx * dy + c * dy * dy))

        return forces_from_energy(energy)          # forces by automatic differentiation


def generate_context():
    system = System([[-0.558, 1.442, 0.0]], numbers=[1], calculator=MullerBrown(), masses=[1.0])
    return Langevin(system, timestep=1.0, temperature=300.0, friction=0.1)     # fs, K, 1/fs


samples = []
def log_cvs(snapshot, state, step):
    samples.append(np.asarray(state.xi).ravel())


cvs = [Component([0], 0), Component([0], 1)]                       # x and y of particle 0
grid = pysages.Grid(lower=(-1.7, -0.5), upper=(1.3, 2.3), shape=(32, 32), periodic=False)
method = SpectralABF(cvs, grid)

result = pysages.run(method, generate_context, TIMESTEPS, callback=log_cvs, stride=20)

analysis = pysages.analyze(result)
mesh, fes = np.asarray(analysis["mesh"]), np.asarray(analysis["free_energy"]).ravel()
np.savetxt("fes.dat", np.c_[mesh, fes - fes.min()], header="x y A_eV")
xi = np.array(samples)
print(f"{len(xi)} CV samples; x in [{xi[:, 0].min():.2f}, {xi[:, 0].max():.2f}], "
      f"y in [{xi[:, 1].min():.2f}, {xi[:, 1].max():.2f}]")
x_min, y_min = mesh[fes.argmin()]
print(f"free-energy minimum at ({x_min:.2f}, {y_min:.2f}); Müller–Brown basin A is at (-0.56, 1.44)")
