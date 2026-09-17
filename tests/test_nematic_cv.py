"""The PySAGES nematic CV must agree with the validated analysis implementation.

Run under an environment that has pysages, jax and ase:
    pytest tests/test_nematic_cv.py
"""
import pathlib
import sys
from typing import NamedTuple

import numpy as np
import pytest
from ase.io import read

# A second PySAGES is installed in this environment
# (/scratch/slv5250/dpmd-ipi/PySAGES) and wins by default, so pin this one.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(1, "/scratch/slv5250/gustavo-lc/code")
sys.path.insert(2, "/scratch/slv5250/gustavo-lc/scripts")

FRAME = "/scratch/slv5250/gustavo-lc/data/sizes/pch5_n792.xyz"
TRAJ = "/scratch/slv5250/gustavo-lc/runs/md/n792/md.extxyz"


class _Data(NamedTuple):
    """What `pysages.colvars.core.build` consumes. `build` jits its `apply`, so
    this has to be a pytree, not a plain object."""

    positions: object
    indices: object


def _snapshot(atoms, positions=None):
    import jax.numpy as jnp

    r = atoms.get_positions() if positions is None else positions
    return _Data(
        jnp.asarray(np.asarray(r, dtype=np.float64)),
        jnp.arange(len(atoms), dtype=jnp.int32),
    )


@pytest.fixture(scope="module")
def frames():
    try:
        return read(TRAJ, index="::400")
    except FileNotFoundError:
        return [read(FRAME)]


def test_matches_the_analysis_implementation(frames):
    from nemora_jax.nematic import NematicOrder
    from pysages.colvars import NematicOrderParameter

    reference = NematicOrder(frames[0])
    cv = NematicOrderParameter.from_atoms(frames[0])
    assert cv.n_molecules == reference.n_molecules
    assert cv.atoms_per_molecule == reference.atoms_per_molecule

    evaluate = cv.function
    for atoms in frames:
        r = np.asarray(atoms.get_positions(), dtype=np.float32)
        assert float(evaluate(r)) == pytest.approx(reference.order(r), abs=2e-3)


def test_matches_the_numpy_reference(frames):
    from nematic_order import frame_order
    from pysages.colvars import NematicOrderParameter

    cv = NematicOrderParameter.from_atoms(frames[0], convention="largest")
    evaluate = cv.function
    for atoms in frames:
        r = np.asarray(atoms.get_positions(), dtype=np.float64)
        assert float(evaluate(r)) == pytest.approx(frame_order(atoms)["S"], abs=2e-3)


def test_is_invariant_under_the_periodic_symmetries(frames):
    from pysages.colvars import NematicOrderParameter

    atoms = frames[0]
    cv = NematicOrderParameter.from_atoms(atoms)
    evaluate = cv.function
    L = np.diag(np.asarray(atoms.get_cell(), dtype=np.float64))
    r = np.asarray(atoms.get_positions(), dtype=np.float64)
    base = float(evaluate(r))

    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert float(evaluate((r @ turn.T) % L)) == pytest.approx(base, abs=1e-5)
    shift = np.array([3.1, -7.4, 11.9])
    assert float(evaluate((r + shift) % L)) == pytest.approx(base, abs=1e-5)


def test_builds_and_differentiates_through_pysages(frames):
    """The path PySAGES actually uses: `build` jits the CV and its Jacobian."""
    import jax.numpy as jnp

    from pysages.colvars import NematicOrderParameter
    from pysages.colvars.core import build

    atoms = frames[0]
    cv = NematicOrderParameter.from_atoms(atoms)
    apply = build(cv, differentiate=True)

    data = _snapshot(atoms)
    xi, jacobian = apply(data)

    assert xi.shape == (1, 1)
    assert jacobian.shape == (1, 3 * len(atoms))
    assert bool(jnp.all(jnp.isfinite(xi)))
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    # a real dependence on the coordinates, not a constant
    assert float(jnp.max(jnp.abs(jacobian))) > 1e-8


def test_finite_difference_matches_the_analytic_gradient(frames):
    import jax.numpy as jnp

    from pysages.colvars import NematicOrderParameter
    from pysages.colvars.core import build

    atoms = frames[0]
    cv = NematicOrderParameter.from_atoms(atoms)
    apply = build(cv, differentiate=True)
    r = np.asarray(atoms.get_positions(), dtype=np.float64)
    _, jacobian = apply(_snapshot(atoms))
    jacobian = np.asarray(jacobian).reshape(-1, 3)

    rng = np.random.default_rng(0)
    probes = rng.choice(len(atoms), size=6, replace=False)
    h = 2e-4
    for i in probes:
        for a in range(3):
            plus, minus = r.copy(), r.copy()
            plus[i, a] += h
            minus[i, a] -= h
            fd = (float(apply(_snapshot(atoms, plus))[0].ravel()[0])
                  - float(apply(_snapshot(atoms, minus))[0].ravel()[0])) / (2 * h)
            assert fd == pytest.approx(jacobian[i, a], abs=2e-3)
