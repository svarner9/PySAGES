<h1 align="center">
  <div>
    <a href="https://github.com/SSAGESLabs/PySAGES" target="_blank">
      <img src="docs/assets/pysages-bottom.svg" alt="PySAGES" height="200">
    </a>
  </div>
  <br>
  <p align="center">
    <a href="https://pysages.readthedocs.io" target="_blank">
      <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Documentation">
    </a>
    &nbsp;
    <a href="https://github.com/SSAGESLabs/PySAGES/actions/workflows/ci.yml" target="_blank">
      <img src="https://github.com/SSAGESLabs/PySAGES/actions/workflows/ci.yml/badge.svg?branch=main" alt="GitHub Actions">
    </a>
    &nbsp;
    <a href="https://github.com/SSAGESLabs/PySAGES/actions/workflows/docker-ci.yml" target="_blank">
      <img src="https://github.com/SSAGESLabs/PySAGES/actions/workflows/docker-ci.yml/badge.svg?branch=main" alt="GitHub Actions">
    </a>
    &nbsp;
    <a href="https://github.com/SSAGESLabs/PySAGES/actions/workflows/trunk.yml" target="_blank">
      <img src="https://github.com/SSAGESLabs/PySAGES/actions/workflows/trunk.yml/badge.svg?branch=main" alt="Trunk">
    </a>
    &nbsp;
    <a href="https://doi.org/10.1038/s41524-023-01189-z" target="_blank">
      <img src="https://img.shields.io/badge/DOI-10.1038%2Fs41524--023--01189--z-blue" alt="Cite PySAGES">
    </a>
  </p>
</h1>

PySAGES (Python Suite for Advanced General Ensemble Simulations) is a Python
implementation of [SSAGES](https://ssagesproject.github.io) with support for GPUs.

## This fork: the `mlmd` backend

This branch adds a backend for [mlmd](https://github.com/svarner9/mlmd) — molecular dynamics with JAX
machine-learning potentials — in `pysages/backends/mlmd.py`. `generate_context()` returns an
`mlmd.Langevin`/`VelocityVerlet` object and the whole biased step (potential, collective variables,
bias, integrator) runs as one compiled JAX program, so enhanced sampling costs about as much as
unbiased MD. See `examples/mlmd/` for the usage pattern and `tests/test_mlmd_backend.py` for the
bias-weight, callback and resume checks. Two small generic fixes ride along: `run()` no longer
mutates the caller's `context_args` dict, and `approxfun.collect_exponents` builds its static index
table with NumPy (required by jax >= 0.10).

### Autoionization collective variables: `CVDist`, `CVNum`

`pysages/colvars/autoionization.py` adds `CVDist` and `CVNum`, a versioned, tested port of the SALT
project's water autoionization CVs (H2O + H2O <-> H3O+ + OH-): `CVNum = sum_O (n_i - 2)^2` (a
coordination-defect sum, `n_i` a rational switching function of O-H distance) and
`CVDist = -sum_{i<k} q_i q_k r_OO` (an environment-weighted O-O distance, `q_i` a per-hydrogen
softmax charge over oxygens). Both take `[oxygen_indices, hydrogen_indices]` and a fixed cell
(NVT only; see the module docstring); salt ions are simply left out of both index groups (or use
`CVDist.from_atoms`/`CVNum.from_atoms` to derive the groups from an `ase.Atoms` frame automatically).
Ported from `/scratch/slv5250/dpmd-ipi/PySAGES/pysages/colvars/{cvNumCollectiveVariableOld,
cvDistCollectiveVariableGem}.py` (the classes `SALT/config_gen/run_biased_md.py` imports), hardened
for float32 (the raw CVnum switching function overflows float32 past `r/R0 ~= 4.87`; see the module
docstring for the clip and its (lack of) effect on the CV value). See `tests/test_autoion_cvs.py` for
value/gradient parity against the reference implementations, float32-vs-float64 agreement, PBC
invariance, and physical sanity checks (intact water, and a constructed H3O+/OH- pair).

## Installation

PySAGES currently supports [HOOMD-blue](https://glotzerlab.engin.umich.edu/hoomd-blue),
[LAMMPS](https://www.lammps.org), [ASE](https://wiki.fysik.dtu.dk/ase/index.html), and
[OpenMM](https://openmm.org). We plan to add support for other engines that have a python
API, including those supported by the original SSAGES.

Before installing PySAGES, you need to install one of the following plugins, depending on
your molecular dynamics engine of choice:

- [DLPack Plugin for HOOMD-blue](https://github.com/SSAGESLabs/hoomd-dlext)
- [DLPack Plugin for OpenMM](https://github.com/SSAGESLabs/openmm-dlext)
- [DLPack Plugin for LAMMPS](https://github.com/SSAGESLabs/lammps-dlext)
- No plugin needed for ASE

You also need to install [JAX](https://github.com/google/jax), a library for
high-performance numerical computing. Follow their installation guide and make sure you
have it installed before using PySAGES. Depending on your local setup, you may need
to install the CPU version or the CUDA compatible version.

For GPU support, JAX and any backend (HOOMD-blue, LAMMPS, or OpenMM) need to be built or
installed with CUDA support.

You can follow our installation tutorial on Google Colab to see how PySAGES, HOOMD-blue,
and OpenMM can be built and installed in such an environment.
[![Colab Badge]][Install Notebook]

## Usage

PySAGES provides a straightforward interface to set up collective variables and enhanced
sampling methods in your MD simulations. See the
[documentation](https://pysages.readthedocs.io/en/latest) to learn more.

We provide ready-to-go examples for common methods. Check out the [examples](examples/)
subfolder to look at different script and notebook examples. These include pre-set
simulations and a tutorial on how to install PySAGES along with the supported MD engines.

## Development

We believe in good software engineering and collaboration. We welcome all contributions.
To ease collaboration, we use [trunk](https://trunk.io) as a development tool free for
open-source software. This includes version-checked linters that can be run automatically.
We ship a launcher for trunk with this repo: `./trunk`, no installation required. For
details about how to use `./trunk fmt` to format existing code into this style and
`./trunk check` to verify a consistent code style, check out the [trunk documentation
page](https://docs.trunk.io/docs).

<!-- References -->

[Colab Badge]: https://colab.research.google.com/assets/colab-badge.svg
[Install Notebook]: https://colab.research.google.com/github/SSAGESLabs/PySAGES/blob/main/examples/Installing_a_PySAGES_Environment.ipynb
