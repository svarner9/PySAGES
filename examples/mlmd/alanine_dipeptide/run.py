#!/usr/bin/env python3
"""Biased MD with a JAX MLIP: SpectralABF over the phi/psi dihedrals of alanine dipeptide.

The MD is mlmd (MACE-OFF23 potential, Langevin thermostat); the sampling method is PySAGES'. Both run
inside one compiled GPU program. Output: fes.dat (free-energy surface), cv.dat (phi, psi vs time),
traj.extxyz and run.log.
"""
import numpy as np
from ase.io import read

import pysages
from pysages.colvars import DihedralAngle
from pysages.methods import SpectralABF

from mlmd import Langevin, Logger, System, Trajectory
from mlmd.calculators import MACE

TIMESTEPS = 200_000          # 200 ps at 1 fs
TEMPERATURE = 300.0          # K
PHI = [1, 3, 4, 6]           # atom indices of the two backbone dihedrals
PSI = [3, 4, 6, 8]


def generate_context():
    """The simulation object, built exactly as for an unbiased mlmd run."""
    atoms = read("ala2.xyz")
    system = System.from_ase(atoms, calculator=MACE("MACE-OFF23_small"))
    system.set_masses({"H": 2.014})                       # deuterate, so 1 fs is a safe timestep
    dyn = Langevin(system, timestep=1.0, temperature=TEMPERATURE, friction=0.01, fix_com=True)
    dyn.attach(Trajectory("traj.extxyz"), interval=500)
    dyn.attach(Logger("run.log", settings=dyn.settings), interval=500)
    return dyn


def log_cvs(snapshot, state, step):
    """Called every `stride` steps with the method state of that frame."""
    phi, psi = np.asarray(state.xi).ravel()
    with open("cv.dat", "a") as fh:
        fh.write(f"{step}\t{phi:.4f}\t{psi:.4f}\n")


cvs = [DihedralAngle(PHI), DihedralAngle(PSI)]
grid = pysages.Grid(lower=(-np.pi, -np.pi), upper=(np.pi, np.pi), shape=(64, 64), periodic=True)
method = SpectralABF(cvs, grid)

result = pysages.run(method, generate_context, TIMESTEPS, callback=log_cvs, stride=100)

analysis = pysages.analyze(result)
mesh, fes = np.asarray(analysis["mesh"]), np.asarray(analysis["free_energy"]).ravel()
np.savetxt("fes.dat", np.c_[mesh, fes - fes.min()], header="phi psi A_eV")
kT = 8.617333262e-5 * TEMPERATURE
print(f"done: free-energy range {(fes.max() - fes.min()) / kT:.1f} kT, written to fes.dat")
