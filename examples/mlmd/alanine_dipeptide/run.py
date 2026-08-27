#!/usr/bin/env python3
"""φ/ψ SpectralABF of alanine dipeptide in vacuum with MACE-OFF23 on the mlmd backend.

    python run.py --geometry ala2.xyz --timesteps 200000 --workdir abf
    python run.py --geometry ala2.xyz --timesteps 200000 --workdir abf      # again: resumes

Writes cv.dat (φ, ψ per frame), traj.extxyz, run.log (mlmd observers) and, at every checkpoint, the
free-energy surface fes.dat plus a resumable result.pickle. Needs mlmd with the MACE stack and a GPU
for useful throughput (~2000 steps/s for this 22-atom molecule on an L40S).
"""
import argparse
import os

import numpy as np
from ase.io import read

import pysages
from pysages.colvars import DihedralAngle
from pysages.methods import SpectralABF
from pysages.methods.core import Result

from mlmd import Langevin, Logger, System, Trajectory
from mlmd.calculators import MACE

KB = 8.617333262e-5   # eV/K


def generate_context(geometry, model, dt, temperature, friction, cueq, workdir, resumed):
    atoms = read(geometry)
    system = System.from_ase(atoms, calculator=MACE(model, cueq=cueq))
    system.set_masses({"H": 2.014})                                  # deuterate: 1 fs is safe
    dyn = Langevin(system, timestep=dt, temperature=temperature, friction=friction, fix_com=True)
    mode = "a" if resumed else "w"
    dyn.attach(Trajectory(os.path.join(workdir, "traj.extxyz"), mode=mode), interval=500)
    dyn.attach(Logger(os.path.join(workdir, "run.log"), settings=dyn.settings, mode=mode), interval=500)
    return dyn


class CVLogger:
    """φ, ψ every frame; free-energy surface + resumable result every ``checkpoint`` frames."""

    def __init__(self, method, workdir, checkpoint=100):
        self.method, self.workdir, self.checkpoint = method, workdir, checkpoint
        self.frames = 0

    def __call__(self, snapshot, state, step):
        with open(os.path.join(self.workdir, "cv.dat"), "a") as fh:
            fh.write(f"{step}\t" + "\t".join(f"{x:.6f}" for x in np.asarray(state.xi).ravel()) + "\n")
        self.frames += 1
        if self.frames % self.checkpoint == 0:
            result = Result(self.method, [state], [self], [snapshot])
            pysages.save(result, os.path.join(self.workdir, "result.pickle"))
            self.write_fes(result)

    def write_fes(self, result):
        analysis = pysages.analyze(result)
        mesh, fes = np.asarray(analysis["mesh"]), np.asarray(analysis["free_energy"]).ravel()
        np.savetxt(os.path.join(self.workdir, "fes.dat"), np.c_[mesh, fes - np.nanmin(fes)],
                   header="phi psi A_eV")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", required=True, help="xyz file of alanine dipeptide")
    ap.add_argument("--phi", type=int, nargs=4, default=[1, 3, 4, 6])
    ap.add_argument("--psi", type=int, nargs=4, default=[3, 4, 6, 8])
    ap.add_argument("--model", default="MACE-OFF23_small")
    ap.add_argument("--cueq", action="store_true")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--friction", type=float, default=0.01)
    ap.add_argument("--workdir", default="abf")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    checkpoint = os.path.join(args.workdir, "result.pickle")
    resumed = os.path.exists(checkpoint)
    context = lambda: generate_context(args.geometry, args.model, args.dt, args.temperature,
                                       args.friction, args.cueq, args.workdir, resumed)
    grid = pysages.Grid(lower=(-np.pi, -np.pi), upper=(np.pi, np.pi), shape=(64, 64), periodic=True)
    method = SpectralABF([DihedralAngle(args.phi), DihedralAngle(args.psi)], grid)
    logger = CVLogger(method, args.workdir)

    if resumed:
        print(f"resuming from {checkpoint}")
        result = pysages.run(pysages.load(checkpoint), context, args.timesteps, stride=args.stride)
    else:
        result = pysages.run(method, context, args.timesteps, callback=logger, stride=args.stride)
    logger.write_fes(result)
    fes = np.loadtxt(os.path.join(args.workdir, "fes.dat"))[:, 2]
    print(f"done: free-energy range {fes.max() / (KB * args.temperature):.1f} kT; see {args.workdir}/")


if __name__ == "__main__":
    main()
