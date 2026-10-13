# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Validation suite for `Funnel_MetaDSpectralABF` and `tq_upper_wall`
(``pysages/methods/funnel_metad_sabf.py``), the ``external_force``/``Frestr`` extension of
`MetaDSpectralABF` (task T2, ``/scratch/slv5250/SALT/plan/metad_sabf_design.md``) requested as a
coordinator follow-up. See that module's docstring for the add-back/``Frestr`` convention these
tests check.

Runs on CPU through the mlmd backend, reusing `test_metad_sabf.py`'s one-particle harness
(``_one_particle``, ``KB``/``T``/``KT``) for the toy tests, and `test_autoion_cvs.py`'s cached
real-water-frame fixtures for the `tq_upper_wall`-specific tests.
"""

import numpy as onp
import pytest

mlmd = pytest.importorskip("mlmd")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pysages  # noqa: E402
from pysages.colvars import Component  # noqa: E402
from pysages.colvars.autoionization import tq_value  # noqa: E402
from pysages.colvars.core import CollectiveVariable, build  # noqa: E402
from pysages.grids import build_indexer  # noqa: E402
from pysages.methods import Funnel_MetaDSpectralABF, MetaDSpectralABF, tq_upper_wall  # noqa: E402
from pysages.methods.funnel_metad_sabf import Funnel_MetaDSpectralABFState  # noqa: E402
from pysages.methods.metad_sabf import MetaDSpectralABFState  # noqa: E402
from pysages.methods.restraints import apply_restraints  # noqa: E402
from pysages.typing import JaxArray, NamedTuple  # noqa: E402

from test_autoion_cvs import FRAMES_PATH, _cell, _groups  # noqa: E402
from test_metad_sabf import KB, KT, T, _one_particle  # noqa: E402

mlmd.enable_float64()


# ---------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------


def _assert_base_fields_equal(a, b, label):
    """Compares every `MetaDSpectralABFState` field of two states (which may themselves be a
    `Funnel_MetaDSpectralABFState`, since its first 19 fields are the base ones in order)."""
    for field in MetaDSpectralABFState._fields:
        va, vb = getattr(a, field), getattr(b, field)
        if field == "fun":  # a `Fun` NamedTuple of ragged (scale, coefficients, c0) leaves
            for sub in va._fields:
                x, y = onp.asarray(getattr(va, sub)), onp.asarray(getattr(vb, sub))
                assert onp.array_equal(x, y), f"{label}: fun.{sub} mismatch"
        else:
            x, y = onp.asarray(va), onp.asarray(vb)
            assert onp.array_equal(x, y), f"{label}: field {field} mismatch"


class _RadialCV(CollectiveVariable):
    """Test-only auxiliary CV: ``r = sqrt(x**2 + y**2)`` of a single atom's position. Used to
    build a generic (non-Tq) `external_force` for the toy tests below, decoupling the
    ``external_force``/``Frestr`` wiring check from `Tq`'s own physics (covered separately by
    the `tq_upper_wall`-specific tests further down)."""

    def __init__(self, indices):
        super().__init__(indices)

    @property
    def function(self):
        return lambda rs: jnp.sqrt(rs[0, 0] ** 2 + rs[0, 1] ** 2 + 1e-12)


def make_radial_wall(indices, threshold, k):
    """A generic upper-wall `external_force` on `_RadialCV`, structured exactly like
    `tq_upper_wall` (same ``apply_restraints`` primitive, same ``eforce = J.T @ f_cv`` sign
    convention -- see that factory's docstring)."""
    radial_cv = _RadialCV(indices)
    apply = build(radial_cv, differentiate=True)
    lo, hi, kl, ku = -jnp.inf, float(threshold), 0.0, float(k)

    def external_force(data):
        r, J_r = apply(data)
        f_cv = apply_restraints(lo, hi, kl, ku, r.reshape(1))
        eforce = (J_r.T @ f_cv).reshape(-1)
        perp = r.reshape(())
        return eforce, perp

    return external_force


class _XYLog:
    """Logs, per step, the main CV (``state.xi``, here x) and the raw y-coordinate of atom 0
    (from ``snapshot``, not otherwise exposed through `MetaDSpectralABFState`) -- everything
    needed to independently recompute the toy radial wall's analytic projected force."""

    def __init__(self):
        self.x, self.y = [], []

    def __call__(self, snapshot, state, step):
        self.x.append(float(onp.asarray(state.xi).ravel()[0]))
        self.y.append(float(onp.asarray(snapshot.positions)[0, 1]))


# ---------------------------------------------------------------------------------------------
# external_force=None reduction
# ---------------------------------------------------------------------------------------------


def test_external_force_none_is_bit_identical_to_metad_sabf():
    """Deliverable requirement: `Funnel_MetaDSpectralABF(..., external_force=None)` must give
    bit-identical results to plain `MetaDSpectralABF` on every field the two share."""

    def energy(positions):
        x = positions[0, 0]
        return 0.5 * 2.0 * x**2 + 0.5 * 2.0 * (positions[0, 1] ** 2 + positions[0, 2] ** 2)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-3.0,), upper=(3.0,), shape=(24,), periodic=False)
    common = dict(
        height=0.01, sigma=0.2, stride=10, ngaussians=400, deltaT=1200.0, kB=KB,
        N=20, fit_threshold=50, fit_freq=25, fit_min_count=10,
    )

    base_result = pysages.run(
        MetaDSpectralABF(cvs, grid, **common), lambda: _one_particle(energy, seed=51), 2000
    )
    funnel_result = pysages.run(
        Funnel_MetaDSpectralABF(cvs, grid, external_force=None, **common),
        lambda: _one_particle(energy, seed=51),
        2000,
    )

    base_state = base_result.states[0]
    funnel_state = funnel_result.states[0]
    _assert_base_fields_equal(base_state, funnel_state, "external_force=None reduction")

    assert onp.all(onp.asarray(funnel_state.Frestr) == 0.0)
    assert onp.all(onp.asarray(funnel_state.Frestr_prod) == 0.0)
    assert float(onp.asarray(funnel_state.perp)) == 0.0

    # positions must also match bit-for-bit: with eforce == 0, the wall contributes nothing to
    # the actual dynamics, so the two trajectories (same seed) must coincide exactly.
    pos_base = onp.asarray(base_result.snapshots[0].positions)
    pos_funnel = onp.asarray(funnel_result.snapshots[0].positions)
    assert onp.array_equal(pos_base, pos_funnel)


# ---------------------------------------------------------------------------------------------
# toy 2D system: a wall on an orthogonal (radial) coordinate
# ---------------------------------------------------------------------------------------------

_TOY_KX = _TOY_KY = _TOY_KZ = 0.1
_TOY_THRESHOLD = 0.5
_TOY_WALL_K = 3.0


def _toy_energy(positions):
    x, y, z = positions[0, 0], positions[0, 1], positions[0, 2]
    return 0.5 * _TOY_KX * x**2 + 0.5 * _TOY_KY * y**2 + 0.5 * _TOY_KZ * z**2


def test_toy_wall_frestr_matches_analytic_projection():
    """Frestr, accumulated per bin as the three-step stencil of ``restr = tsolve(Jxi, eforce)``
    from the previous three updates, must match an
    INDEPENDENTLY reconstructed analytic projection built from a per-step log of the raw
    trajectory (x, y) and the known closed form of the wall + main CV Jacobians -- not merely
    "small"/"zero", a genuine nonzero, hand-computable quantity (the main CV's Jacobian, a
    one-hot at x, picks out exactly the wall's x-component: ``restr = (x/r) * f_wall(r)``, where
    ``f_wall(r) = k*(r-threshold)`` for ``r > threshold`` and 0 otherwise -- see
    `make_radial_wall`/`tq_upper_wall`'s shared sign convention)."""

    def ctx():
        return _one_particle(_toy_energy, mass=1.0, dt=0.5, friction=0.3, seed=61)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-2.5,), upper=(2.5,), shape=(24,), periodic=False)
    wall = make_radial_wall([0], threshold=_TOY_THRESHOLD, k=_TOY_WALL_K)
    method = Funnel_MetaDSpectralABF(
        cvs, grid, height=0.0, sigma=0.2, stride=1000000, ngaussians=2, deltaT=None,
        N=10**9, fit_threshold=10**9, fit_freq=10**9, fit_min_count=10**9,
        external_force=wall,
    )
    log = _XYLog()
    result = pysages.run(method, ctx, 20000, callback=log, stride=1)

    x = onp.array(log.x)
    y = onp.array(log.y)
    r = onp.sqrt(x**2 + y**2 + 1e-12)
    f_wall = onp.where(r > _TOY_THRESHOLD, _TOY_WALL_K * (r - _TOY_THRESHOLD), 0.0)
    analytic_restr = (x / r) * f_wall
    assert onp.any(f_wall > 0.0), "wall never fired -- toy parameters need adjusting"
    assert onp.any(f_wall == 0.0), "wall always fired -- toy parameters need adjusting"

    # Each sample is credited with the three-step stencil of the external force applied at the
    # three PREVIOUS updates (stencil_addback default; see funnel_metad_sabf.py), r_{-1..-3} = 0.
    def lag(a, k):
        return onp.concatenate([onp.zeros(k), a[:-k]])

    stencil_restr = (2.25 * lag(analytic_restr, 1) - 1.5 * lag(analytic_restr, 2)
                     + 0.25 * lag(analytic_restr, 3))
    get_index = build_indexer(method.grid)
    nbins = int(method.grid.shape[0])
    expected = onp.zeros(nbins)
    for xv, rv in zip(x, stencil_restr):
        idx = int(onp.asarray(get_index(jnp.array([xv])))[0])
        if 0 <= idx < nbins:
            expected[idx] += rv

    actual = onp.asarray(result.states[0].Frestr).reshape(-1)
    assert onp.allclose(expected, actual, atol=1e-6, rtol=1e-6)
    # And the burn_in=0 default means Frestr_prod == Frestr exactly.
    assert onp.array_equal(onp.asarray(result.states[0].Frestr_prod).reshape(-1), actual)


@pytest.mark.slow
def test_toy_wall_restrained_mean_force_matches_exact_restricted_ensemble():
    """The RESTRAINED mean force / FES (`Funnel_MetaDSpectralABF`'s inherited ``mean_force``,
    which per the module docstring adds back ONLY the method's own bias, never ``eforce``) must
    match the EXACT conditional mean force of the wall-restricted ensemble,
    ``<F_x | x> = [int dy F_x(x,y) w(x,y)] / [int dy w(x,y)]`` with
    ``w(x,y) = exp(-beta*(0.5*ky*y**2 + U_wall(sqrt(x**2+y**2))))``, computed here by direct 1D
    numerical quadrature over y (independent of the MD/estimator code entirely)."""
    beta = 1.0 / KT

    def exact_mean_force(x, y_max=6.0, ny=8001):
        y = onp.linspace(-y_max, y_max, ny)
        r = onp.sqrt(x**2 + y**2)
        above = r > _TOY_THRESHOLD
        Uwall = onp.where(above, 0.5 * _TOY_WALL_K * (r - _TOY_THRESHOLD) ** 2, 0.0)
        dUwall_dr = onp.where(above, _TOY_WALL_K * (r - _TOY_THRESHOLD), 0.0)
        safe_r = onp.maximum(r, 1e-12)
        Fx = -_TOY_KX * x - dUwall_dr * (x / safe_r)
        w = onp.exp(-beta * (0.5 * _TOY_KY * y**2 + Uwall))
        trapz = getattr(onp, "trapezoid", None) or onp.trapz
        return trapz(Fx * w, y) / trapz(w, y)

    def ctx():
        return _one_particle(_toy_energy, mass=1.0, dt=0.5, friction=0.3, seed=62)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-2.5,), upper=(2.5,), shape=(32,), periodic=False)
    wall = make_radial_wall([0], threshold=_TOY_THRESHOLD, k=_TOY_WALL_K)
    method = Funnel_MetaDSpectralABF(
        cvs, grid, height=0.0, sigma=0.2, stride=1000000, ngaussians=2, deltaT=None,
        N=50, fit_threshold=10**9, fit_freq=10**9, fit_min_count=10**9,
        external_force=wall,
    )
    result = pysages.run(method, ctx, 300000)
    ana = pysages.analyze(result)

    mesh = onp.asarray(ana["mesh"]).reshape(-1)
    hist = onp.asarray(ana["histogram"]).reshape(-1)
    mean_force = onp.asarray(ana["mean_force"]).reshape(-1)
    exact = onp.array([exact_mean_force(xm) for xm in mesh])

    used = hist > 200
    err = mean_force[used] - exact[used]
    rel_rms = float(onp.sqrt(onp.mean(err**2)) / onp.mean(onp.abs(exact[used])))
    print(
        f"[toy wall restrained mean force] bins used {used.sum()}/{len(mesh)}; "
        f"relative RMS error: {rel_rms:.3f}; max_restraint_force={ana['max_restraint_force']:.4f}"
    )
    assert onp.isfinite(mean_force).all()
    assert rel_rms < 0.3


# ---------------------------------------------------------------------------------------------
# tq_upper_wall on a real 128-water frame
# ---------------------------------------------------------------------------------------------


class _FrameData(NamedTuple):
    positions: JaxArray
    indices: JaxArray


@pytest.fixture(scope="module")
def frames():
    if not FRAMES_PATH.exists():
        pytest.skip(f"cached test frames not found: {FRAMES_PATH}")
    import ase.io

    return ase.io.read(str(FRAMES_PATH), index=":")


def test_tq_wall_zero_below_and_finite_difference_above(frames):
    atoms = frames[0]
    oxygen, hydrogen = _groups(atoms)
    cell = _cell(atoms)
    pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
    natoms = len(atoms)
    data = _FrameData(jnp.asarray(pos), jnp.arange(natoms, dtype=jnp.int32))

    p_o64 = jnp.asarray(pos[oxygen])
    p_h64 = jnp.asarray(pos[hydrogen])
    tq_now = float(tq_value(p_o64, p_h64, jnp.asarray(cell)))

    # (1) below threshold: exactly zero force (bit-exact, per `apply_restraints`'s `where`).
    wall_below = tq_upper_wall(oxygen.tolist(), hydrogen.tolist(), cell, tq_now + 1.0, k=3.0)
    eforce_below, perp_below = wall_below(data)
    assert float(perp_below) == pytest.approx(tq_now, rel=1e-10)
    assert bool(jnp.all(eforce_below == 0.0))

    # (2) above threshold: nonzero, and matching a finite-difference gradient check.
    threshold_above = tq_now - 0.3
    k = 3.0
    wall_above = tq_upper_wall(oxygen.tolist(), hydrogen.tolist(), cell, threshold_above, k=k)
    eforce_above, perp_above = wall_above(data)
    assert float(perp_above) == pytest.approx(tq_now, rel=1e-10)
    assert bool(jnp.any(eforce_above != 0.0))

    def wall_potential(p_o, p_h):
        tq = tq_value(p_o, p_h, jnp.asarray(cell))
        return 0.5 * k * jnp.maximum(tq - threshold_above, 0.0) ** 2

    g_o, g_h = jax.grad(wall_potential, argnums=(0, 1))(p_o64, p_h64)
    atom_force = onp.asarray((-eforce_above).reshape(natoms, 3))  # actual force applied
    assert onp.allclose(atom_force[oxygen], onp.asarray(-g_o), atol=1e-6)
    assert onp.allclose(atom_force[hydrogen], onp.asarray(-g_h), atol=1e-6)

    # central finite differences at a few random atom/axis probes (mirrors
    # test_tq_and_aux_wall.py::test_tq_gradient_matches_finite_differences).
    rng = onp.random.default_rng(7)
    h = 1e-5
    worst = 0.0
    for _ in range(6):
        group, count = (0, len(oxygen)) if rng.random() < 0.5 else (1, len(hydrogen))
        idx, axis = int(rng.integers(0, count)), int(rng.integers(0, 3))
        p_o_p, p_h_p = onp.array(pos[oxygen]), onp.array(pos[hydrogen])
        (p_o_p if group == 0 else p_h_p)[idx, axis] += h
        plus = float(wall_potential(jnp.asarray(p_o_p), jnp.asarray(p_h_p)))
        p_o_m, p_h_m = onp.array(pos[oxygen]), onp.array(pos[hydrogen])
        (p_o_m if group == 0 else p_h_m)[idx, axis] -= h
        minus = float(wall_potential(jnp.asarray(p_o_m), jnp.asarray(p_h_m)))
        fd = (plus - minus) / (2 * h)
        analytic = float((g_o if group == 0 else g_h)[idx, axis])
        worst = max(worst, abs(fd - analytic))
    print(f"[Tq wall gradient] worst |analytic - central FD|: {worst:.3e}")
    assert worst < 1e-4


def test_tq_wall_float32_matches_float64(frames):
    worst = 0.0
    for atoms in frames[:6]:
        oxygen, hydrogen = _groups(atoms)
        pos = onp.asarray(atoms.get_positions(), dtype=onp.float64)
        natoms = len(atoms)
        cell64 = _cell(atoms)

        tq_now = float(
            tq_value(jnp.asarray(pos[oxygen]), jnp.asarray(pos[hydrogen]), jnp.asarray(cell64))
        )
        threshold = tq_now - 0.3

        data64 = _FrameData(jnp.asarray(pos), jnp.arange(natoms, dtype=jnp.int32))
        wall64 = tq_upper_wall(oxygen.tolist(), hydrogen.tolist(), cell64, threshold, k=3.0)
        eforce64, _ = wall64(data64)

        data32 = _FrameData(
            jnp.asarray(pos, dtype=jnp.float32), jnp.arange(natoms, dtype=jnp.int32)
        )
        cell32 = jnp.asarray(cell64, dtype=jnp.float32)
        wall32 = tq_upper_wall(
            oxygen.tolist(), hydrogen.tolist(), cell32,
            onp.float32(threshold), k=onp.float32(3.0),
        )
        eforce32, _ = wall32(data32)

        denom = max(float(jnp.max(jnp.abs(eforce64))), 1e-8)
        rel = float(jnp.max(jnp.abs(eforce64 - eforce32))) / denom
        worst = max(worst, rel)
    print(f"[Tq wall float32 vs float64] worst relative diff over frames: {worst:.3e}")
    assert worst < 1e-2


# ---------------------------------------------------------------------------------------------
# pickling / serialization and bit-exact resume
# ---------------------------------------------------------------------------------------------


def test_pickle_round_trip_with_external_force(tmp_path):
    """The `external_force` closure itself must survive a `pysages.save`/`pysages.load`
    round-trip (via `dill`, through `self.kwargs` -- see the module docstring's discussion of
    why this mirrors `Funnel_MetaDABF`/`Funnel_SpectralABF`'s identical, already-established
    pattern), and the reloaded method must keep applying it identically."""

    def ctx():
        return _one_particle(_toy_energy, mass=1.0, dt=0.5, friction=0.3, seed=71)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-2.5,), upper=(2.5,), shape=(24,), periodic=False)
    wall = make_radial_wall([0], threshold=_TOY_THRESHOLD, k=_TOY_WALL_K)
    method = Funnel_MetaDSpectralABF(
        cvs, grid, height=0.01, sigma=0.2, stride=10, ngaussians=300, deltaT=1200.0, kB=KB,
        N=20, fit_threshold=50, fit_freq=25, fit_min_count=10, external_force=wall,
    )
    result = pysages.run(method, ctx, 300)
    p = tmp_path / "funnel.pickle"
    pysages.save(result, p)
    loaded = pysages.load(p)

    assert callable(loaded.method.kwargs.get("external_force"))
    for field in Funnel_MetaDSpectralABFState._fields:
        va = getattr(result.states[0], field)
        vb = getattr(loaded.states[0], field)
        if field == "fun":
            for sub in va._fields:
                assert onp.array_equal(
                    onp.asarray(getattr(va, sub)), onp.asarray(getattr(vb, sub))
                ), f"pickle round-trip: fun.{sub} mismatch"
        else:
            assert onp.array_equal(onp.asarray(va), onp.asarray(vb)), (
                f"pickle round-trip: field {field} mismatch"
            )

    one_more = pysages.run(loaded, ctx, 1)
    another = pysages.run(result, ctx, 1)
    _assert_base_fields_equal(one_more.states[0], another.states[0], "one post-pickle update")
    assert onp.array_equal(
        onp.asarray(one_more.states[0].Frestr), onp.asarray(another.states[0].Frestr)
    )


def test_restart_is_bit_exact_with_the_backend_fix(tmp_path):
    """Backend fix c069399 (memo Sec. 5): mlmd `Sampler.restore` restores `step`/`key`, so a
    split run (save + load + resume) must match a single continuous run bit-for-bit -- both the
    MD trajectory and every accumulator, including the new `Frestr`."""
    k, mass = 2.0, 1.0

    def energy(positions):
        x = positions[0, 0]
        return 0.5 * k * x**2 + 0.5 * k * (positions[0, 1] ** 2 + positions[0, 2] ** 2)

    def ctx():
        return _one_particle(energy, mass=mass, dt=0.5, friction=0.3, seed=72)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-2.0,), upper=(2.0,), shape=(20,), periodic=False)
    wall = make_radial_wall([0], threshold=0.6, k=3.0)

    def make_method():
        return Funnel_MetaDSpectralABF(
            cvs, grid, height=0.005, sigma=0.1, stride=20, ngaussians=200, deltaT=1200.0, kB=KB,
            N=10, fit_threshold=50, fit_freq=25, external_force=wall,
        )

    cont = pysages.run(make_method(), ctx, 800)

    part1 = pysages.run(make_method(), ctx, 400)
    p = tmp_path / "funnel_restart.pickle"
    pysages.save(part1, p)
    part2 = pysages.run(pysages.load(p), ctx, 400)

    pos_cont = onp.asarray(cont.snapshots[0].positions)
    pos_split = onp.asarray(part2.snapshots[0].positions)
    hist_cont = onp.asarray(cont.states[0].hist)
    hist_split = onp.asarray(part2.states[0].hist)
    Frestr_cont = onp.asarray(cont.states[0].Frestr)
    Frestr_split = onp.asarray(part2.states[0].Frestr)
    assert onp.array_equal(pos_cont, pos_split)
    assert onp.array_equal(hist_cont, hist_split)
    assert onp.array_equal(Frestr_cont, Frestr_split)


# ---------------------------------------------------------------------------------------------
# analyze() surface
# ---------------------------------------------------------------------------------------------


def test_analyze_has_restraint_keys_and_diagnostic():
    def ctx():
        return _one_particle(_toy_energy, mass=1.0, dt=0.5, friction=0.3, seed=81)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-2.5,), upper=(2.5,), shape=(24,), periodic=False)
    wall = make_radial_wall([0], threshold=_TOY_THRESHOLD, k=_TOY_WALL_K)
    method = Funnel_MetaDSpectralABF(
        cvs, grid, height=0.005, sigma=0.2, stride=25, ngaussians=200, deltaT=1200.0, kB=KB,
        N=20, fit_threshold=500, fit_freq=100, fit_min_count=20, external_force=wall,
    )
    result = pysages.run(method, ctx, 8000)
    ana = pysages.analyze(result)

    for key in (
        "histogram", "mean_force", "free_energy", "mesh", "fun", "fes_fn",
        "free_energy_trapz", "metad_potential", "heights", "n_clipped",
        "restraint_mean_force", "restraint_energy", "corrected_free_energy",
        "max_restraint_force",
    ):
        assert key in ana, f"missing analyze() key: {key}"

    assert "force_corrected" not in ana  # see the module docstring's autofes combine.py.j2 note
    assert onp.isfinite(onp.asarray(ana["restraint_mean_force"])).all()
    assert onp.isfinite(onp.asarray(ana["restraint_energy"])).all()
    assert onp.isfinite(onp.asarray(ana["corrected_free_energy"])).all()
    assert onp.isfinite(ana["max_restraint_force"])

    # `tq_upper_wall`'s float32-vs-float64 numerical parity is covered directly by
    # `test_tq_wall_float32_matches_float64` above; toggling the mlmd/jax process-wide x64 flag
    # mid-session here (to also exercise the whole method loop in float32) would risk destabilizing
    # every other float64-assuming test in this session, so it is deliberately not done -- see the
    # module docstring.
