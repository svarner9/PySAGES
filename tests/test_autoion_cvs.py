# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Value/gradient parity and physical sanity tests for the autoionization CVs
``CVDist``/``CVNum`` (``pysages/colvars/autoionization.py``), ported for the
SALT project's pKw campaign (task T1 of ``SALT/plan/PAPER_PLAN.md``).

Reference implementations checked against (read-only; see
``pysages/colvars/autoionization.py`` module docstring for the full
provenance discussion):

  - ``/scratch/slv5250/DFT/autofes/examples/ase_2d/cvs/{cvdist,cvnum}.py``
    (a hardened copy of the SALT project's original CVs: adds the CVnum
    clip/singularity guard this port also uses).
  - ``/scratch/slv5250/dpmd-ipi/PySAGES/pysages/colvars/
    {cvNumCollectiveVariableOld,cvDistCollectiveVariableGem}.py`` -- the
    actual classes ``CvNumCollectiveVariableOld``/``CvDistCollectiveVariableGem``
    that ``SALT/config_gen/run_biased_md.py`` imports by name (see that
    script's own header comment, which names this exact path). These two
    files are untracked in that checkout's git history (``git status`` there
    lists them as ``??``), which is why the ``pysages`` wheels pip-installed
    into the ``base``/``analysis`` conda environments (built from a
    git-tracked snapshot) do not contain them.

Test data: 29 real frames cached at ``tests/data/autoion_cv_frames.extxyz``,
pulled from ``/scratch/slv5250/SALT/data/ml/mace_fmm_3581/train.extxyz``
(category in ``atoms.info["category"]``): 8 category-C pure-water,
6 category-C with a dissolved salt (CaI2/LiI/CaCl2), 6 category-Cb pure-water,
6 category-Cb with salt, and 3 category-A (bulk equilibrium, i.e. generically
intact water, no constructed ionization) frames. See
``/scratch/slv5250/tmp/claude-*/scratchpad/extract_frames.py`` (not part of
this repo) for the exact selection script if the cache ever needs
regenerating from a newer corpus.
"""

import importlib.util
import pathlib

import ase.io
import jax
import numpy as onp
import pytest
from jax import numpy as jnp

from pysages.colvars import CVDist, CVNum
from pysages.colvars.autoionization import (
    DEFAULT_LAMBDA,
    DEFAULT_R0,
    cv_dist_value,
    cv_num_value,
)
from pysages.colvars.core import build
from pysages.typing import JaxArray, NamedTuple

HERE = pathlib.Path(__file__).resolve().parent
FRAMES_PATH = HERE / "data" / "autoion_cv_frames.extxyz"

REF_CVDIST_ASE2D = pathlib.Path("/scratch/slv5250/DFT/autofes/examples/ase_2d/cvs/cvdist.py")
REF_CVNUM_ASE2D = pathlib.Path("/scratch/slv5250/DFT/autofes/examples/ase_2d/cvs/cvnum.py")
REF_CVNUM_OLD = pathlib.Path(
    "/scratch/slv5250/dpmd-ipi/PySAGES/pysages/colvars/cvNumCollectiveVariableOld.py"
)
REF_CVDIST_GEM = pathlib.Path(
    "/scratch/slv5250/dpmd-ipi/PySAGES/pysages/colvars/cvDistCollectiveVariableGem.py"
)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _groups(atoms):
    symbols = atoms.get_chemical_symbols()
    oxygen = onp.array([i for i, s in enumerate(symbols) if s == "O"])
    hydrogen = onp.array([i for i, s in enumerate(symbols) if s == "H"])
    return oxygen, hydrogen


def _cell(atoms):
    return onp.asarray(atoms.get_cell()[:], dtype=onp.float64)


def _discrete_ion_pair(atoms, oxygen, hydrogen, free_cut=1.30):
    """``probes/md_autoionization.py``-style discrete assignment: each H goes
    to its nearest O (MIC); a single O with exactly 1 H (OH-) and a single O
    with exactly 3 H (H3O+), no unclaimed H, identify a clean ion pair.
    Returns the O-O MIC distance, or ``None`` if the frame does not show a
    single clean pair (e.g. a delocalized/Grotthuss-smeared state).

    Assumes an orthorhombic cell (as ``md_autoionization.py`` itself does);
    all frames in this test's cache satisfy this.
    """
    cell = _cell(atoms)
    box = onp.diag(cell)
    pos = atoms.get_positions()
    d = pos[hydrogen][:, None, :] - pos[oxygen][None, :, :]
    d -= box * onp.round(d / box)
    r = onp.linalg.norm(d, axis=2)
    owner = onp.argmin(r, axis=1)
    dmin = r[onp.arange(len(hydrogen)), owner]
    if (dmin > free_cut).any():
        return None
    counts = onp.bincount(owner, minlength=len(oxygen))
    donors = onp.where(counts == 1)[0]
    acceptors = onp.where(counts == 3)[0]
    if len(donors) != 1 or len(acceptors) != 1:
        return None
    doo = pos[oxygen[acceptors[0]]] - pos[oxygen[donors[0]]]
    doo -= box * onp.round(doo / box)
    return float(onp.linalg.norm(doo))


def _make_isolated_pair(separation, box=30.0, bond_length=0.98):
    """A single, isolated H3O+ (3 H) / OH- (1 H) pair, O-O separation
    ``separation`` along x, far from every periodic image (``box`` large).
    No other water: this isolates the CV's "clean ion pair" limit from any
    background bulk-water softmax cross-talk."""
    cell = onp.eye(3) * box
    o_h3o = onp.array([box / 2, box / 2, box / 2])
    directions = onp.array([[1, 0, 0.3], [-0.5, 0.87, 0.3], [-0.5, -0.87, 0.3]], dtype=float)
    directions /= onp.linalg.norm(directions, axis=1, keepdims=True)
    h3o_hydrogens = o_h3o + bond_length * directions
    o_oh = o_h3o + onp.array([separation, 0.0, 0.0])
    oh_direction = onp.array([1.0, 0.2, 0.0])
    oh_direction /= onp.linalg.norm(oh_direction)
    oh_hydrogen = o_oh + bond_length * oh_direction
    p_o = onp.stack([o_h3o, o_oh])
    p_h = onp.vstack([h3o_hydrogens, oh_hydrogen[None, :]])
    return jnp.asarray(p_o), jnp.asarray(p_h), jnp.asarray(cell)


@pytest.fixture(scope="module")
def frames():
    if not FRAMES_PATH.exists():
        pytest.skip(f"cached test frames not found: {FRAMES_PATH}")
    return ase.io.read(str(FRAMES_PATH), index=":")


@pytest.fixture(scope="module")
def reference_modules():
    modules = {}
    for path, name in (
        (REF_CVDIST_ASE2D, "ref_cvdist_ase2d"),
        (REF_CVNUM_ASE2D, "ref_cvnum_ase2d"),
        (REF_CVNUM_OLD, "ref_cvnum_old"),
        (REF_CVDIST_GEM, "ref_cvdist_gem"),
    ):
        if not path.exists():
            pytest.skip(f"reference implementation not found: {path}")
        modules[name] = _load_module(path, name)
    return modules


# --------------------------------------------------------------------------
# (a) value parity vs the reference implementations, float64, on real frames
# --------------------------------------------------------------------------


def test_cvdist_matches_ase2d_reference(frames, reference_modules):
    ref = reference_modules["ref_cvdist_ase2d"]
    worst = 0.0
    for atoms in frames:
        oxygen, hydrogen = _groups(atoms)
        cell = jnp.asarray(_cell(atoms))
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        p_o, p_h = jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen])
        mine = float(cv_dist_value(p_o, p_h, cell, DEFAULT_LAMBDA))
        theirs = float(ref.compute_cv_dist(p_o, p_h, cell, DEFAULT_LAMBDA))
        rel = abs(mine - theirs) / max(abs(theirs), 1e-12)
        worst = max(worst, rel)
        assert rel < 1e-6, f"{atoms.info.get('frame_id')}: mine={mine} ref={theirs}"
    print(f"[CVDist vs autofes/ase_2d reference] worst relative diff: {worst:.3e}")


def test_cvnum_matches_ase2d_reference(frames, reference_modules):
    ref = reference_modules["ref_cvnum_ase2d"]
    worst = 0.0
    for atoms in frames:
        oxygen, hydrogen = _groups(atoms)
        cell = jnp.asarray(_cell(atoms))
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        p_o, p_h = jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen])
        mine = float(cv_num_value(p_o, p_h, cell, DEFAULT_R0))
        theirs = float(ref.compute_cv_num(p_o, p_h, cell, DEFAULT_R0))
        rel = abs(mine - theirs) / max(abs(theirs), 1e-12)
        worst = max(worst, rel)
        assert rel < 1e-6, f"{atoms.info.get('frame_id')}: mine={mine} ref={theirs}"
    print(f"[CVNum vs autofes/ase_2d reference] worst relative diff: {worst:.3e}")


def test_cvdist_matches_the_production_gem_class(frames, reference_modules):
    """Parity with `CvDistCollectiveVariableGem`, the class SALT's
    `run_biased_md.py` actually imports at runtime."""
    ref = reference_modules["ref_cvdist_gem"]
    worst = 0.0
    for atoms in frames:
        oxygen, hydrogen = _groups(atoms)
        cell = jnp.asarray(_cell(atoms))
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        p_o, p_h = jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen])
        mine = float(cv_dist_value(p_o, p_h, cell, DEFAULT_LAMBDA))
        theirs = float(ref._compute_cv_dist(p_o, p_h, cell, DEFAULT_LAMBDA))
        rel = abs(mine - theirs) / max(abs(theirs), 1e-12)
        worst = max(worst, rel)
        assert rel < 1e-6, f"{atoms.info.get('frame_id')}: mine={mine} ref={theirs}"
    print(f"[CVDist vs production CvDistCollectiveVariableGem] worst relative diff: {worst:.3e}")


def test_cvnum_matches_the_production_old_class(frames, reference_modules):
    """Parity with `CvNumCollectiveVariableOld`, the class SALT's
    `run_biased_md.py` actually imports at runtime. That class has neither
    the clip nor the singularity guard (see the module docstring); the two
    agree to float64 precision on real frames because clipping only ever
    touches O-H pairs whose contribution is already ~0."""
    ref = reference_modules["ref_cvnum_old"]
    worst = 0.0
    for atoms in frames:
        oxygen, hydrogen = _groups(atoms)
        cell = jnp.asarray(_cell(atoms))
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        p_o, p_h = jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen])
        mine = float(cv_num_value(p_o, p_h, cell, DEFAULT_R0))
        theirs = float(ref.compute_cv_num(p_o, p_h, cell, DEFAULT_R0))
        rel = abs(mine - theirs) / max(abs(theirs), 1e-12)
        worst = max(worst, rel)
        assert rel < 1e-6, f"{atoms.info.get('frame_id')}: mine={mine} ref={theirs}"
    print(f"[CVNum vs production CvNumCollectiveVariableOld] worst relative diff: {worst:.3e}")


# --------------------------------------------------------------------------
# (b) gradient parity: jax.grad vs central finite differences
# --------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["CVDist", "CVNum"])
def test_gradient_matches_finite_differences(frames, which):
    fn = cv_dist_value if which == "CVDist" else cv_num_value
    param = DEFAULT_LAMBDA if which == "CVDist" else DEFAULT_R0
    rng = onp.random.default_rng(0)
    h = 1e-5
    worst = 0.0
    # A handful of frames x a handful of probe coordinates keeps this fast
    # while still exercising O-H, O-O and cross terms across real geometries.
    for atoms in (frames[2], frames[9], frames[-1]):
        oxygen, hydrogen = _groups(atoms)
        cell = jnp.asarray(_cell(atoms))
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        p_o = jnp.asarray(pos[oxygen])
        p_h = jnp.asarray(pos[hydrogen])
        g_o, g_h = jax.grad(fn, argnums=(0, 1))(p_o, p_h, cell, param)
        for _ in range(6):
            group, count = (0, len(oxygen)) if rng.random() < 0.5 else (1, len(hydrogen))
            idx, axis = rng.integers(0, count), rng.integers(0, 3)
            p_o_p, p_h_p = onp.array(pos[oxygen]), onp.array(pos[hydrogen])
            target = p_o_p if group == 0 else p_h_p
            target[idx, axis] += h
            plus = float(fn(jnp.asarray(p_o_p), jnp.asarray(p_h_p), cell, param))
            p_o_m, p_h_m = onp.array(pos[oxygen]), onp.array(pos[hydrogen])
            target = p_o_m if group == 0 else p_h_m
            target[idx, axis] -= h
            minus = float(fn(jnp.asarray(p_o_m), jnp.asarray(p_h_m), cell, param))
            fd = (plus - minus) / (2 * h)
            analytic = float((g_o if group == 0 else g_h)[idx, axis])
            worst = max(worst, abs(fd - analytic))
    print(f"[{which} gradient] worst |analytic - central FD| over probes: {worst:.3e}")
    assert worst < 1e-4


def test_gradient_finite_through_pysages_build(frames):
    """The path PySAGES actually uses: `build` jits CVDist/CVNum and their
    Jacobian via `jax.grad`, exactly as a sampling method would call them."""

    class _Data(NamedTuple):
        positions: JaxArray
        indices: JaxArray

    atoms = frames[0]
    oxygen, hydrogen = _groups(atoms)
    cell = _cell(atoms)
    cv_dist = CVDist([oxygen.tolist(), hydrogen.tolist()], cell)
    cv_num = CVNum([oxygen.tolist(), hydrogen.tolist()], cell)

    pos = jnp.asarray(onp.asarray(atoms.get_positions(), dtype=onp.float64))
    data = _Data(pos, jnp.arange(len(atoms), dtype=jnp.int32))

    for cv in (cv_dist, cv_num):
        apply = build(cv, differentiate=True)
        xi, jacobian = apply(data)
        assert xi.shape == (1, 1)
        assert jacobian.shape == (1, 3 * len(atoms))
        assert bool(jnp.all(jnp.isfinite(xi)))
        assert bool(jnp.all(jnp.isfinite(jacobian)))
        assert float(jnp.max(jnp.abs(jacobian))) > 1e-8


# --------------------------------------------------------------------------
# (c) float32 vs float64 agreement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["CVDist", "CVNum"])
def test_float32_matches_float64(frames, which):
    fn = cv_dist_value if which == "CVDist" else cv_num_value
    param = DEFAULT_LAMBDA if which == "CVDist" else DEFAULT_R0
    worst = 0.0
    for atoms in frames:
        oxygen, hydrogen = _groups(atoms)
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        cell64 = jnp.asarray(_cell(atoms))
        v64 = float(fn(jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen]), cell64, param))

        cell32 = jnp.asarray(_cell(atoms), dtype=jnp.float32)
        p_o32 = jnp.asarray(pos[oxygen], dtype=jnp.float32)
        p_h32 = jnp.asarray(pos[hydrogen], dtype=jnp.float32)
        v32 = float(fn(p_o32, p_h32, cell32, onp.float32(param)))

        rel = abs(v64 - v32) / max(abs(v64), 1e-8)
        worst = max(worst, rel)
    print(
        f"[{which} float32 vs float64] worst relative diff over {len(frames)} frames: {worst:.3e}"
    )
    # Real measured worst case on this cache is O(1e-6)-O(1e-7) (small,
    # 128-water systems); the wide margin below documents rather than
    # papers over the cancelling-sum float32 risk flagged in
    # SALT/config_gen/run_biased_md.py -- production biasing should still
    # keep the CV machinery in float64.
    assert worst < 1e-2


# --------------------------------------------------------------------------
# (d) PBC: translation / per-atom re-imaging invariance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["CVDist", "CVNum"])
def test_pbc_invariance(frames, which):
    fn = cv_dist_value if which == "CVDist" else cv_num_value
    param = DEFAULT_LAMBDA if which == "CVDist" else DEFAULT_R0
    rng = onp.random.default_rng(1)
    worst = 0.0
    for atoms in (frames[1], frames[8], frames[-2]):
        oxygen, hydrogen = _groups(atoms)
        cell_np = _cell(atoms)
        cell = jnp.asarray(cell_np)
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        base = float(fn(jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen]), cell, param))

        # Independently re-image every atom by an arbitrary lattice vector:
        # a much stronger test than a single rigid shift, since it exercises
        # the MIC image choice per atom rather than a globally common one.
        shifts = rng.integers(-3, 4, size=(pos.shape[0], 3))
        shifted = pos + shifts @ cell_np
        moved = float(fn(jnp.asarray(shifted[oxygen]), jnp.asarray(shifted[hydrogen]), cell, param))
        worst = max(worst, abs(base - moved))
    print(f"[{which} PBC invariance] worst |base - shifted|: {worst:.3e}")
    assert worst < 1e-6


# --------------------------------------------------------------------------
# (e) sanity: intact water, and a constructed H3O+/OH- pair
# --------------------------------------------------------------------------


def test_intact_water_is_near_zero(frames):
    """Category-A (bulk equilibrium) frames are generic MD snapshots, not
    constructed ionization events: both CVs should sit close to their
    "intact water" value of 0."""
    intact = [a for a in frames if a.info.get("category") == "A"]
    assert intact, "expected at least one category-A frame in the test cache"
    for atoms in intact:
        oxygen, hydrogen = _groups(atoms)
        cell = jnp.asarray(_cell(atoms))
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        p_o, p_h = jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen])
        cvd = float(cv_dist_value(p_o, p_h, cell, DEFAULT_LAMBDA))
        cvn = float(cv_num_value(p_o, p_h, cell, DEFAULT_R0))
        print(f"[intact water] {atoms.info.get('frame_id')}: CVDist={cvd:.4f} CVNum={cvn:.4f}")
        assert abs(cvd) < 0.1
        assert 0.0 <= cvn < 0.3


@pytest.mark.parametrize("separation", [3.0, 4.0, 5.0, 7.0, 9.0])
def test_isolated_ion_pair_cvdist_tracks_oo_distance(separation):
    """A hand-built, isolated H3O+/OH- pair (no bulk water diluting the
    per-H softmax) at O-O distance `separation` should give CVDist ~=
    separation: each ion's charge q_i saturates to (nearly) +-1 when there
    is only one competing oxygen, so -q_i q_k r_OO ~= -(+1)(-1) r_OO = r_OO."""
    p_o, p_h, cell = _make_isolated_pair(separation)
    cvd = float(cv_dist_value(p_o, p_h, cell, DEFAULT_LAMBDA))
    print(f"[isolated pair] d={separation:.2f} CVDist={cvd:.4f}")
    assert abs(cvd - separation) < 0.02


def test_isolated_ion_pair_close_contact():
    """At short range (near the CIP regime) the two ions' softmaxes start
    mixing a bit more, so allow a looser tolerance than the well-separated
    cases above; CVDist should still closely track the O-O distance."""
    separation = 2.6
    p_o, p_h, cell = _make_isolated_pair(separation)
    cvd = float(cv_dist_value(p_o, p_h, cell, DEFAULT_LAMBDA))
    print(f"[isolated pair, close contact] d={separation:.2f} CVDist={cvd:.4f}")
    assert abs(cvd - separation) < 0.1


def test_c_category_ladder_vs_discrete_assignment(frames):
    """For each category-C (constructed proton-wire) frame that shows a
    single, clean H3O+/OH- pair under the `probes/md_autoionization.py`
    nearest-owner rule, report CVDist alongside that discrete O-O
    separation and check they are in the same ballpark. CVDist is a soft,
    continuous CV (its ion charges never fully saturate to +-1, especially
    with bulk water and/or salt around), so it undershoots the hard
    discrete distance somewhat -- this is expected, not a bug, and is why
    the check below is a loose ratio rather than a tight absolute one.

    Frames without a single clean pair (`None`) are Grotthuss-delocalized
    or multi-defect states where the discrete assignment does not apply;
    they are reported but not checked quantitatively.
    """
    c_frames = [a for a in frames if a.info.get("category") == "C"]
    assert c_frames, "expected category-C frames in the test cache"
    checked = 0
    for atoms in c_frames:
        oxygen, hydrogen = _groups(atoms)
        cell = jnp.asarray(_cell(atoms))
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        p_o, p_h = jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen])
        cvd = float(cv_dist_value(p_o, p_h, cell, DEFAULT_LAMBDA))
        discrete = _discrete_ion_pair(atoms, oxygen, hydrogen)
        salt = atoms.info.get("salt", "water")
        print(
            f"[C-ladder] salt={salt:8s} sep_OO(build)={atoms.info.get('sep_OO')!s:>8s} "
            f"discrete_OO={discrete} CVDist={cvd:.4f}"
        )
        if discrete is None:
            continue
        checked += 1
        ratio = cvd / discrete
        assert 0.4 <= ratio <= 1.1, f"CVDist/discrete_OO={ratio:.3f} out of range for {salt}"
    assert checked >= 3, "expected several C-category frames with a clean discrete ion pair"
