# PySAGES + mlmd

[mlmd](../../../mlmd) runs molecular dynamics with JAX machine-learning potentials, with the whole
MD step compiled into one GPU program. With this backend the PySAGES sampling method (its collective
variables, bias and estimator updates) runs *inside* that program, so ABF, metadynamics, umbrella
sampling, … cost about as much as unbiased MD — thousands of steps per second for a small molecule.

The pattern is the same as for the other backends: `generate_context()` returns the simulation
object, here an `mlmd.Langevin` (or `VelocityVerlet`) built exactly as for an unbiased run:

```python
import pysages
from pysages.colvars import DihedralAngle
from pysages.methods import SpectralABF
from mlmd import System, Langevin
from mlmd.calculators import MACE

def generate_context():
    system = System.from_ase(atoms, calculator=MACE("MACE-OFF23_small"))
    return Langevin(system, timestep=0.5, temperature=300, friction=0.01)   # fs, K, 1/fs

method = SpectralABF([DihedralAngle([1, 3, 4, 6]), DihedralAngle([3, 4, 6, 8])], grid)
result = pysages.run(method, generate_context, 200_000, callback=my_logger, stride=100)
```

Anything attached to the dynamics (`dyn.attach(Trajectory(...), interval=...)`) is still written,
and any calculator mlmd accepts — MACE, Lennard-Jones, an analytic surface, your own JAX potential —
works unchanged. `stride` is how often (in MD steps) the Python `callback(snapshot, state, step)` is
invoked; `frames_per_chunk` (default 10) how many such frames are batched per device call.

| example | what it shows | needs |
|---|---|---|
| `muller_brown/` | SpectralABF on an analytic 2-D surface: the complete pattern on one screen, CPU, a minute | jax, mlmd, pysages |
| `alanine_dipeptide/` | φ/ψ SpectralABF of alanine dipeptide with MACE-OFF23, trajectory + log output | + the MACE stack (`mlmd/install.sh`) and a GPU |

Run them from their own directory: `cd muller_brown && python run.py` (on the cluster:
`MLMD_PYSAGES=/path/to/this/checkout /path/to/mlmd/env/mlmd.sh python run.py`). Settings are the
constants at the top of each script. To continue a run later, `pysages.save(result, "result.pickle")`
and then `pysages.run(pysages.load("result.pickle"), generate_context, more_steps)`.

## Checked against the ASE backend

`alanine_dipeptide/run.py` for 1 M steps (MACE-OFF23-small, 1 fs, 300 K) reproduces the φ/ψ
free-energy surface of a 1 M-step PySAGES + ASE + mace-torch run of the same system: RMSD 1.8 kT and
Pearson r = 0.96 over the well-sampled region, with the C7eq, C5 and αR minima at the same
positions to within 0.2 rad and 0.6 kT (the remaining difference is the usual ABF extrapolation into
unsampled regions plus the finite sampling of both runs). Wall time: 22 minutes on an RTX PRO 6000
including model loading and compilation, versus ~3 hours for the ASE reference on an L40S. The
exact bias-weight test lives in `tests/test_mlmd_backend.py`.
