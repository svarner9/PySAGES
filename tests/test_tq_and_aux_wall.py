# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Tests for the ``Tq`` collective variable (``pysages/colvars/autoionization.py``) and the
``AuxiliaryWall`` composable restraint (``pysages/methods/aux_restraint.py``), added on top of
task T2 per a coordinator follow-up request (see ``/scratch/slv5250/SALT/plan/metad_sabf_design.md``
and PAPER_PLAN's SALT global-CV discussion).

Physics reference for ``Tq``: Calegari Andrade et al., PNAS 2023, Eq. 4, ``Tq = sum_o
sqrt(q_o**2 + c)`` from the same per-oxygen Grifoni charges ``CVDist`` uses. Note on the
"intact water -> 0" claim: that is exact only for ``c -> 0`` and only up to the
``N_O * sqrt(c)`` baseline this smoothing introduces (every intact-water oxygen contributes
``sqrt(0 + c) = sqrt(c)``, not exactly 0). For a small synthetic system (2 oxygens, as in the
isolated-pair fixture below) that baseline is negligible next to the ~2 signal from a real pair;
for a real multi-water box it is not exactly negligible in absolute terms, so this test checks
intact frames against the *theoretical fully-associated baseline* ``n_oxygen * sqrt(c)``
(equivalent to every ``q_o ~= 0``) rather than against 0 outright.
"""

import ase.io
import jax
import numpy as onp
import pytest
from jax import numpy as jnp

from pysages.colvars import Component, Tq
from pysages.colvars.autoionization import DEFAULT_LAMBDA, DEFAULT_TQ_C, tq_value
from pysages.colvars.core import build
from pysages.methods.funnel_metad_sabf import (
    MEASURED_ONE_PAIR_TQ_INCREMENT,
    calibrate_tq_threshold,
    tq_baseline,
    tq_single_pair_threshold,
)
from pysages.typing import JaxArray, NamedTuple

from test_autoion_cvs import (  # noqa: E402
    FRAMES_PATH,
    _cell,
    _discrete_ion_pair,
    _groups,
    _make_isolated_pair,
    _make_two_isolated_pairs,
)


@pytest.fixture(scope="module")
def frames():
    if not FRAMES_PATH.exists():
        pytest.skip(f"cached test frames not found: {FRAMES_PATH}")
    return ase.io.read(str(FRAMES_PATH), index=":")


# --------------------------------------------------------------------------
# value / gradient parity
# --------------------------------------------------------------------------


def test_tq_gradient_matches_finite_differences(frames):
    rng = onp.random.default_rng(1)
    h = 1e-5
    worst = 0.0
    for atoms in (frames[2], frames[9], frames[-1]):
        oxygen, hydrogen = _groups(atoms)
        cell = jnp.asarray(_cell(atoms))
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        p_o = jnp.asarray(pos[oxygen])
        p_h = jnp.asarray(pos[hydrogen])
        g_o, g_h = jax.grad(tq_value, argnums=(0, 1))(p_o, p_h, cell, DEFAULT_LAMBDA, DEFAULT_TQ_C)
        for _ in range(6):
            group, count = (0, len(oxygen)) if rng.random() < 0.5 else (1, len(hydrogen))
            idx, axis = rng.integers(0, count), rng.integers(0, 3)
            p_o_p, p_h_p = onp.array(pos[oxygen]), onp.array(pos[hydrogen])
            target = p_o_p if group == 0 else p_h_p
            target[idx, axis] += h
            plus = float(tq_value(jnp.asarray(p_o_p), jnp.asarray(p_h_p), cell))
            p_o_m, p_h_m = onp.array(pos[oxygen]), onp.array(pos[hydrogen])
            target = p_o_m if group == 0 else p_h_m
            target[idx, axis] -= h
            minus = float(tq_value(jnp.asarray(p_o_m), jnp.asarray(p_h_m), cell))
            fd = (plus - minus) / (2 * h)
            analytic = float((g_o if group == 0 else g_h)[idx, axis])
            worst = max(worst, abs(fd - analytic))
    print(f"[Tq gradient] worst |analytic - central FD| over probes: {worst:.3e}")
    assert worst < 1e-4


def test_tq_float32_matches_float64(frames):
    worst = 0.0
    for atoms in frames:
        oxygen, hydrogen = _groups(atoms)
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        cell64 = jnp.asarray(_cell(atoms))
        v64 = float(tq_value(jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen]), cell64))

        cell32 = jnp.asarray(_cell(atoms), dtype=jnp.float32)
        p_o32 = jnp.asarray(pos[oxygen], dtype=jnp.float32)
        p_h32 = jnp.asarray(pos[hydrogen], dtype=jnp.float32)
        v32 = float(
            tq_value(p_o32, p_h32, cell32, onp.float32(DEFAULT_LAMBDA), onp.float32(DEFAULT_TQ_C))
        )
        rel = abs(v64 - v32) / max(abs(v64), 1e-8)
        worst = max(worst, rel)
    print(f"[Tq float32 vs float64] worst relative diff over {len(frames)} frames: {worst:.3e}")
    assert worst < 1e-3


def test_tq_gradient_finite_through_pysages_build(frames):
    class _Data(NamedTuple):
        positions: JaxArray
        indices: JaxArray

    atoms = frames[0]
    oxygen, hydrogen = _groups(atoms)
    cell = _cell(atoms)
    cv = Tq([oxygen.tolist(), hydrogen.tolist()], cell)

    pos = jnp.asarray(onp.asarray(atoms.get_positions(), dtype=onp.float64))
    data = _Data(pos, jnp.arange(len(atoms), dtype=jnp.int32))

    apply = build(cv, differentiate=True)
    xi, jacobian = apply(data)
    assert xi.shape == (1, 1)
    assert jacobian.shape == (1, 3 * len(atoms))
    assert bool(jnp.all(jnp.isfinite(xi)))
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    assert float(jnp.max(jnp.abs(jacobian))) > 1e-8


# --------------------------------------------------------------------------
# physical sanity
# --------------------------------------------------------------------------


def test_tq_intact_water_near_fully_associated_baseline(frames):
    """See the module docstring: intact water sits near `n_oxygen * sqrt(c)`, the value every
    q_o ~= 0 gives, not near 0 outright."""
    intact = [a for a in frames if a.info.get("category") == "A"]
    assert intact, "expected at least one category-A frame in the test cache"
    for atoms in intact:
        oxygen, hydrogen = _groups(atoms)
        cell = jnp.asarray(_cell(atoms))
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        p_o, p_h = jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen])
        tq = float(tq_value(p_o, p_h, cell))
        baseline = len(oxygen) * onp.sqrt(DEFAULT_TQ_C)
        print(f"[intact water] {atoms.info.get('frame_id')}: Tq={tq:.4f} baseline={baseline:.4f}")
        assert abs(tq - baseline) < 0.1 * max(baseline, 1.0)


def test_tq_isolated_ion_pair_is_near_two():
    p_o, p_h, cell = _make_isolated_pair(separation=5.0)
    tq = float(tq_value(p_o, p_h, cell))
    baseline = p_o.shape[0] * onp.sqrt(DEFAULT_TQ_C)  # 2 oxygens here: 1 H3O+, 1 OH-
    print(f"[isolated pair] Tq={tq:.4f} (baseline for 2 oxygens: {baseline:.4f})")
    assert abs(tq - 2.0) < 0.3


# --------------------------------------------------------------------------
# tq_single_pair_threshold / calibrate_tq_threshold: the calibration-bug fix
# (2026-09-25) -- see funnel_metad_sabf.py's "CALIBRATION BUG" module note.
# The bug: the old calibrate_tq_threshold(tq_trajectory, percentile=99.9, margin=0.05),
# calibrated on an UNBIASED intact-water trajectory, produced a threshold (51.350603 for 512
# oxygens, see runs/stability_0925/tq_threshold.json) BELOW the Tq level of even a single
# admitted ion pair -- it would forbid the very pair `tq_upper_wall` exists to admit. These
# tests check the fix admits every single-pair fixture this test module has (the isolated
# constructed pair at several separations, plus every real category-C/Cb frame with a clean
# discrete single pair) and rejects a constructed two-pair frame; the old, un-migrated
# `calibrate_tq_threshold` call signature (no `one_pair_increment`) must now raise instead of
# silently returning an unsafe threshold.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("separation", [2.6, 3.0, 4.0, 5.0, 7.0, 9.0])
def test_tq_single_pair_threshold_admits_isolated_pair_across_separations(separation):
    p_o, p_h, cell = _make_isolated_pair(separation)
    tq = float(tq_value(p_o, p_h, cell))
    threshold = tq_single_pair_threshold(n_oxygen=p_o.shape[0])
    print(f"[single-pair threshold, isolated] sep={separation:.2f} tq={tq:.4f} "
          f"threshold={threshold:.4f}")
    assert tq < threshold


def test_tq_single_pair_threshold_admits_real_single_pair_frames(frames):
    """Every real category-C/Cb frame in the cache that shows exactly one clean H3O+/OH- pair
    under the discrete nearest-owner assignment (same rule as `salt_md.defects`, via
    `_discrete_ion_pair`) must sit below `tq_single_pair_threshold` for that frame's oxygen
    count -- the property the old, buggy calibration violated (see the module note above)."""
    n_checked = 0
    worst_margin = float("inf")
    for atoms in frames:
        if atoms.info.get("category") not in ("C", "Cb"):
            continue
        oxygen, hydrogen = _groups(atoms)
        if _discrete_ion_pair(atoms, oxygen, hydrogen) is None:
            continue
        cell = jnp.asarray(_cell(atoms))
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        p_o, p_h = jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen])
        tq = float(tq_value(p_o, p_h, cell))
        threshold = tq_single_pair_threshold(n_oxygen=len(oxygen))
        margin = threshold - tq
        worst_margin = min(worst_margin, margin)
        salt = atoms.info.get("salt", "water")
        print(f"[single-pair threshold, real] {atoms.info.get('frame_id')} salt={salt} "
              f"tq={tq:.4f} threshold={threshold:.4f} margin={margin:.4f}")
        assert tq < threshold, f"{atoms.info.get('frame_id')}: tq={tq} >= threshold={threshold}"
        n_checked += 1
    print(f"[single-pair threshold, real] checked {n_checked} frames, worst margin "
          f"{worst_margin:.4f}")
    assert n_checked >= 3, "expected several real single-clean-pair C/Cb frames in the cache"


def test_tq_single_pair_threshold_rejects_constructed_two_pair_frame():
    p_o, p_h, cell = _make_two_isolated_pairs(separation=5.0)
    tq = float(tq_value(p_o, p_h, cell))
    threshold = tq_single_pair_threshold(n_oxygen=p_o.shape[0])
    print(f"[two-pair frame] tq={tq:.4f} threshold={threshold:.4f}")
    assert tq > threshold


def test_tq_single_pair_threshold_between_one_and_two_pair_levels():
    """Directly checks the documented formula: threshold sits at the midpoint (default
    margin=0.5) between baseline+increment and baseline+2*increment."""
    n_oxygen = 512
    threshold = tq_single_pair_threshold(n_oxygen)
    baseline = tq_baseline(n_oxygen)
    one_pair = baseline + MEASURED_ONE_PAIR_TQ_INCREMENT
    two_pair = baseline + 2 * MEASURED_ONE_PAIR_TQ_INCREMENT
    assert one_pair < threshold < two_pair
    assert threshold == pytest.approx((one_pair + two_pair) / 2, rel=1e-12)


def test_calibrate_tq_threshold_raises_without_one_pair_increment():
    """The core bug fix: the old call signature (no `one_pair_increment`) must now be a hard
    error, not a silently-too-low threshold (see the module's "CALIBRATION BUG" note)."""
    trajectory = [51.213, 51.223, 51.227, 51.306]  # runs/stability_0925/tq_threshold.json-like
    with pytest.raises(ValueError, match="one_pair_increment"):
        calibrate_tq_threshold(trajectory)


def test_calibrate_tq_threshold_with_increment_admits_a_real_single_pair_level(frames):
    """With `one_pair_increment` supplied, `calibrate_tq_threshold` on the ACTUAL buggy
    trajectory shape (unbiased 512-oxygen intact water, baseline ~51.2, spread ~51.21-51.31)
    now produces a threshold safely above a real single-pair Tq level scaled to 512 oxygens --
    the exact failure the old signature could not catch."""
    tq_trajectory = [51.213006852516, 51.223060599990255, 51.30589324783915, 51.22797774142322]
    threshold = calibrate_tq_threshold(tq_trajectory, MEASURED_ONE_PAIR_TQ_INCREMENT)
    assert threshold > max(tq_trajectory) + MEASURED_ONE_PAIR_TQ_INCREMENT

    # a real single-pair frame's Tq, rescaled from its own (small) box to 512 oxygens by
    # keeping the same measured increment over an analytic 512-oxygen baseline, must sit below
    # this empirically-calibrated threshold too.
    real_single_pair = None
    for atoms in frames:
        if atoms.info.get("category") not in ("C", "Cb"):
            continue
        oxygen, hydrogen = _groups(atoms)
        if _discrete_ion_pair(atoms, oxygen, hydrogen) is None:
            continue
        cell = jnp.asarray(_cell(atoms))
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        p_o, p_h = jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen])
        tq = float(tq_value(p_o, p_h, cell))
        increment = tq - tq_baseline(len(oxygen))
        real_single_pair = tq_baseline(512) + increment
        break
    assert real_single_pair is not None, "expected at least one real single-pair C/Cb frame"
    assert real_single_pair < threshold


# --------------------------------------------------------------------------
# AuxiliaryWall: composable upper/lower wall on a CV a method never grids
# --------------------------------------------------------------------------


def test_aux_wall_zero_below_threshold_and_restoring_above():
    """Direct check of `apply_restraints`'s convention, as `AuxiliaryWall` uses it: exactly the
    property the coordinator asked for ("the wall exerts zero force below the threshold")."""
    from pysages.methods.restraints import apply_restraints

    lo, hi, kl, ku = -jnp.inf, 2.2, 0.0, jnp.asarray(100.0 / 96.485)  # 100 kJ/mol -> eV
    below = apply_restraints(lo, hi, kl, ku, jnp.array([1.5]))
    at = apply_restraints(lo, hi, kl, ku, jnp.array([2.2]))
    above = apply_restraints(lo, hi, kl, ku, jnp.array([2.5]))
    assert float(below[0]) == 0.0
    assert float(at[0]) == 0.0
    assert float(above[0]) == pytest.approx(float(ku) * 0.3, rel=1e-10)
    assert float(above[0]) > 0.0  # positive -> `bias = -J^T f` restores xi back toward the wall


def test_aux_wall_composes_with_a_gridded_method_via_bias_addition():
    """`AuxiliaryWall` never touches the wrapped method's own state layout or grid: with the aux
    CV safely below its threshold the combined bias must equal the wrapped method's bias alone
    (the wall contributes exactly zero); moving the aux CV above threshold must add a nonzero
    restoring force. Uses a synthetic two-atom "snapshot" (a plain `Component` on atom 1's x as
    a stand-in for `Tq`) so this needs no real water frame or mlmd context -- it only exercises
    the wrapper's plumbing against the exact `HelperMethods`/`build()` contract other backends
    (mlmd included) use.
    """
    import numpy as np

    from pysages.backends.snapshot import HelperMethods
    from pysages.colvars import Component as _Comp
    from pysages.methods import AuxiliaryWall, HarmonicBias
    from pysages.typing import JaxArray as _JA

    class _Data(NamedTuple):
        positions: _JA
        indices: _JA
        momenta: _JA

    class _FakeSnapshot(NamedTuple):
        positions: _JA
        vel_mass: tuple
        forces: _JA
        ids: _JA
        box: object
        dt: float

    class _FakeBox(NamedTuple):
        H: _JA
        origin: _JA

    natoms = 2
    ids = jnp.arange(natoms)
    box = _FakeBox(jnp.eye(3), jnp.zeros(3))

    def make_snapshot(x1):
        positions = jnp.zeros((natoms, 3)).at[1, 0].set(x1)
        return _FakeSnapshot(positions, (jnp.zeros((natoms, 3)), jnp.ones((natoms, 1))),
                              jnp.zeros((natoms, 3)), ids, box, 1.0)

    def query(snapshot):
        return _Data(snapshot.positions, ids, jnp.zeros((natoms, 3)).flatten())

    def dimensionality():
        return 3

    helpers = HelperMethods(query, dimensionality)

    base = HarmonicBias([_Comp([0], 0)], kspring=5.0, center=[0.0])
    aux_cv = _Comp([1], 0)  # stand-in for Tq: atom 1's x-coordinate
    wrapped = AuxiliaryWall(base, aux_cv, threshold=1.0, k=2.0, side="upper")

    snap0 = make_snapshot(0.5)  # aux CV below threshold
    _, base_init, base_update = base.build(snap0, helpers)
    _, wrapped_init, wrapped_update = wrapped.build(snap0, helpers)

    base_below = base_update(snap0, base_init())
    aux_below = wrapped_update(snap0, wrapped_init())
    assert np.allclose(np.asarray(aux_below.bias), np.asarray(base_below.bias), atol=1e-10)

    snap1 = make_snapshot(2.0)  # aux CV above threshold: 2*k*(2.0-1.0) = 2.0 extra force on atom 1
    aux_above = wrapped_update(snap1, wrapped_init())
    assert float(aux_above.bias[1, 0]) < 0.0  # restoring: pushes atom 1 back toward -x
    assert not np.allclose(np.asarray(aux_above.bias[1]), 0.0)
