# Müller-Brown with ABF

Smallest possible biased run: one particle on the 2D Müller-Brown potential, SpectralABF on its
x and y coordinates. Runs on a laptop CPU in about a minute. Nothing here needs a GPU or an MLIP,
so it's the place to check that PySAGES and mlmd talk to each other before spending GPU time.

`python run.py`

## What's in run.py

The potential. `MullerBrown` is a complete mlmd calculator: `r_max = None` because there is no
neighbor list to build, and `build()` hands back a function of (positions, neighbors) that returns
energy and forces. The forces come from `jax.grad`, so only the energy is written out. Swap this
class for `MACE(...)` (or your own) and the rest of the script doesn't change.

The dynamics. `generate_context()` builds a `System` with one atom and returns a `Langevin`
integrator. That's the same object you'd `.run()` for plain MD; PySAGES takes it as is.

The method. Two `Component` CVs (x and y of atom 0), a 32x32 grid, `SpectralABF`. `pysages.run`
does the work; `log_cvs` gets called every 20 steps with the CV values.

## What to expect

```
10000 CV samples; x in [-2.80, 1.39], y in [-0.78, 2.89]
free-energy minimum at (-0.56, 1.50); Müller–Brown basin A is at (-0.56, 1.44)
```

The particle starts in basin A and the bias pushes it over both saddles within the 200k steps, so
the CV range covers the whole grid. The minimum sits on the deepest well to within a grid spacing.
`fes.dat` has three columns (x, y, free energy in eV, min-subtracted); plot it with any heatmap
tool. Barriers in the scaled potential are about 8 kT, so the surface spans roughly 35 kT once the
high corners of the grid are counted.
