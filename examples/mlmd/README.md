# PySAGES with mlmd

mlmd runs MD with JAX machine-learning potentials and compiles the whole step into one GPU
program. With this backend the PySAGES method (CVs, bias, estimator) runs inside that program, so
ABF, metadynamics or umbrella sampling cost about what unbiased MD costs.

Usage is the same as with the other backends. `generate_context()` returns the simulation object,
which here is an `mlmd.Langevin` (or `VelocityVerlet`) built exactly as you would for an unbiased
run:

```python
def generate_context():
    system = System.from_ase(atoms, calculator=MACE("MACE-OFF23_small"))
    return Langevin(system, timestep=1.0, temperature=300, friction=0.01)   # fs, K, 1/fs

method = SpectralABF([DihedralAngle([1, 3, 4, 6]), DihedralAngle([3, 4, 6, 8])], grid)
result = pysages.run(method, generate_context, 200_000, callback=log_cvs, stride=100)
```

Anything attached to the dynamics (`Trajectory`, `Logger`) still gets written, and any calculator
mlmd accepts works: MACE, Lennard-Jones, an analytic surface, your own JAX potential. `stride` is
how often, in MD steps, the callback is called.

Two examples, each with its own README:

- `muller_brown/` - the complete pattern on one screen, analytic potential, CPU, about a minute.
- `alanine_dipeptide/` - MACE-OFF23, phi/psi ABF, GPU, a few minutes.

Run them from their own directory (`cd muller_brown && python run.py`). On the group cluster:
`MLMD_PYSAGES=/path/to/this/checkout /path/to/mlmd/env/mlmd.sh python run.py`.

The backend itself is `pysages/backends/mlmd.py`; `tests/test_mlmd_backend.py` checks that the bias
enters with the right weight (a harmonic bias on a harmonic well gives exactly kT/(k + k_bias)), that
callbacks and observers fire when they should, and that a saved run resumes.
