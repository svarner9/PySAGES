# Alanine dipeptide with MACE and ABF

The real thing: a MACE-OFF23 potential, deuterated alanine dipeptide in vacuum, SpectralABF over
the backbone dihedrals phi and psi. Needs the MACE stack (see mlmd's install.sh) and a GPU;
200k steps take a few minutes on an L40S or better.

`python run.py`

## What's in run.py

Settings at the top: number of steps, temperature, the atom indices of the two dihedrals for the
geometry in ala2.xyz.

`generate_context()` is the unbiased setup you'd write for mlmd anyway: read the xyz, attach the
`MACE` calculator, deuterate so 1 fs is safe, make a `Langevin` with the center of mass pinned, and
attach a trajectory writer and a log. PySAGES calls this once.

`log_cvs` is the PySAGES callback. It runs every 100 steps (the `stride` in `pysages.run`) and
appends phi and psi to cv.dat.

The last block sets up the CVs, the 64x64 periodic grid and the method, runs, and writes the
free-energy surface.

## What to expect

Output files: traj.extxyz and run.log (from mlmd, every 500 steps), cv.dat (phi, psi every 100
steps) and fes.dat (phi, psi, free energy in eV, min-subtracted).

After 200k steps the surface already shows the usual alanine dipeptide basins. In our runs the
minimum is C7eq at about phi = -1.5, psi = 1.3 (radians), with C5 near (-2.7, 2.9) at ~1 kT and the
alpha-R region near (-1.5, 0.6) at ~2 kT. The printed free-energy range is around 30 kT. A 1M step
run of this script matches a 1M step PySAGES + ASE + mace-torch run of the same system to about
2 kT RMSD over the sampled region, with the same minima.

Throughput is roughly 2000 steps/s on an L40S including the ABF work, against ~90 steps/s for the
ASE backend with mace-torch on the same GPU. To run longer, raise TIMESTEPS; to continue a run,
`pysages.save(result, "result.pickle")` at the end and later
`pysages.run(pysages.load("result.pickle"), generate_context, more_steps, callback=log_cvs, stride=100)`.
