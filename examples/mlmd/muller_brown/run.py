#!/usr/bin/env python3
"""SpectralABF on the Müller–Brown surface with the mlmd backend — the complete pattern, no MLIP needed.

A single "atom" moves on a 2-D analytic potential written as a 10-line mlmd calculator; the
collective variables are its x and y coordinates. Runs in seconds on a CPU.
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

import pysages
from pysages.colvars import Component
from pysages.methods import SpectralABF

from mlmd import Langevin, System
from mlmd.calculators import forces_from_energy

# Müller–Brown parameters; `scale` puts the A -> B barrier at ~8 kT for T = 300 K
A = jnp.array([-200.0, -100.0, -170.0, 15.0])
a = jnp.array([-1.0, -1.0, -6.5, 0.7])
b = jnp.array([0.0, 0.0, 11.0, 0.6])
c = jnp.array([-10.0, -10.0, -6.5, 0.7])
x0 = jnp.array([1.0, 0.0, -0.5, -1.0])
y0 = jnp.array([0.0, 0.5, 1.5, 1.0])


class MullerBrown:
    """An mlmd calculator: ``r_max=None`` (no neighbor list), ``build`` returns ``ef(R, neighbors)``."""

    r_max = None
    dtype = jnp.float32

    def __init__(self, scale=0.002):
        self.scale = scale

    def build(self, numbers, cell=None, pbc=(False, False, False)):
        def energy(positions, neighbors):
            dx, dy = positions[0, 0] - x0, positions[0, 1] - y0
            return self.scale * jnp.sum(A * jnp.exp(a * dx * dx + b * dx * dy + c * dy * dy))

        return forces_from_energy(energy)          # forces by automatic differentiation


def generate_context():
    system = System([[-0.558, 1.442, 0.0]], numbers=[1], calculator=MullerBrown(), masses=[1.0])
    return Langevin(system, timestep=1.0, temperature=300.0, friction=0.1, seed=0)   # fs, K, 1/fs


class Logger:
    def __init__(self):
        self.xi = []

    def __call__(self, snapshot, state, step):
        self.xi.append(np.asarray(state.xi).ravel())


def main(timesteps):
    cvs = [Component([0], 0), Component([0], 1)]                     # x and y of particle 0
    grid = pysages.Grid(lower=(-1.7, -0.5), upper=(1.3, 2.3), shape=(32, 32), periodic=False)
    method = SpectralABF(cvs, grid)
    logger = Logger()
    result = pysages.run(method, generate_context, timesteps, callback=logger, stride=20)
    analysis = pysages.analyze(result)
    mesh, fes = np.asarray(analysis["mesh"]), np.asarray(analysis["free_energy"]).ravel()
    xi = np.array(logger.xi)
    kT = 8.617333262e-5 * 300.0
    print(f"{len(xi)} CV samples; x in [{xi[:, 0].min():.2f}, {xi[:, 0].max():.2f}], "
          f"y in [{xi[:, 1].min():.2f}, {xi[:, 1].max():.2f}]")
    print(f"free-energy minimum at (x, y) = {tuple(np.round(mesh[np.nanargmin(fes)], 2))} "
          f"(Müller–Brown basin A is at (-0.56, 1.44)); range {(np.nanmax(fes) - np.nanmin(fes)) / kT:.1f} kT")
    np.savetxt("fes.dat", np.c_[mesh, fes - np.nanmin(fes)], header="x y A_eV")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    main(parser.parse_args().timesteps)
