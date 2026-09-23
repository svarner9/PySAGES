# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Validation suite for `MetaDSpectralABF` (task T2,
``/scratch/slv5250/SALT/plan/metad_sabf_design.md`` Sec. 6) and the mlmd backend resume fix
(Sec. 5) it depends on. Section numbers in test docstrings refer to that memo.

Runs on CPU through the mlmd backend with small analytic `Calculator`s (one particle; harmonic
y/z confinement where the memo's harmonic-y/z convention applies). Most tests are fast (a few
seconds); the 1D convergence toys (``@pytest.mark.slow``) run 3-6e5 steps each (~15-40s apiece
without a per-step callback) and are meant to run once and be read for their printed numbers,
not as a tight pass/fail gate on FES shape.
"""

import numpy as onp
import pytest

mlmd = pytest.importorskip("mlmd")

import jax.numpy as jnp  # noqa: E402
import pysages  # noqa: E402
from mlmd import Langevin, System, units  # noqa: E402
from mlmd.calculators import Harmonic  # noqa: E402
from mlmd.calculators.base import _CalculatorBase, forces_from_energy  # noqa: E402
from pysages.colvars import Component  # noqa: E402
from pysages.methods import ABF, MetaDSpectralABF, SpectralABF  # noqa: E402
from pysages.methods.metad_sabf import build_gaussian_accumulator, build_metad_interpolator  # noqa: E402

mlmd.enable_float64()

KB = units.kB  # eV/K
T = 300.0
KT = KB * T


class Analytic(_CalculatorBase):
    """Wraps an arbitrary jax energy function ``energy(positions) -> scalar`` as an mlmd
    `Calculator`, for toy potentials with an exactly known free energy along one CV."""

    def __init__(self, energy_fn, dtype="float64"):
        super().__init__(dtype)
        self._energy_fn = energy_fn

    def build(self, numbers, cell=None, pbc=(False, False, False)):
        fn = self._energy_fn

        def energy(positions, neighbors):
            return fn(positions)

        return forces_from_energy(energy)


def _one_particle(energy_fn, mass=1.0, dt=0.5, temperature=T, friction=0.5, seed=0):
    system = System(onp.zeros((1, 3)), [2], Analytic(energy_fn, dtype="float64"), masses=[mass])
    return Langevin(system, timestep=dt, temperature=temperature, friction=friction, seed=seed)


class CVLog:
    def __init__(self):
        self.steps, self.xi, self.Wp, self.force_applied = [], [], [], []

    def __call__(self, snapshot, state, step):
        self.steps.append(step)
        self.xi.append(float(onp.asarray(state.xi).ravel()[0]))
        self.Wp.append(float(onp.asarray(state.Wp).ravel()[0]))
        self.force_applied.append(float(onp.asarray(state.force_applied).ravel()[0]))


def _bin_average(x, values, edges):
    nbins = len(edges) - 1
    idx = onp.clip(onp.digitize(x, edges) - 1, 0, nbins - 1)
    s = onp.zeros(nbins)
    c = onp.zeros(nbins)
    good = onp.isfinite(values)
    onp.add.at(s, idx[good], onp.asarray(values)[good])
    onp.add.at(c, idx[good], 1)
    return onp.where(c > 0, s / onp.maximum(c, 1), onp.nan), c


# ---------------------------------------------------------------------------------------------
# 6.1a -- metad subtraction exactness, with a negative control that omits the add-back
# ---------------------------------------------------------------------------------------------


def test_metad_addback_is_exact_vs_negative_control():
    """Sec. 6.1a: oversized, non-well-tempered metad (h0 = 2 kT, stride 10) on a separable 1D
    harmonic well, abf_bias=False so the sABF fit never touches the estimate. The class's own
    `Fsum`/`hist` already add back `force_applied`; this test independently reconstructs BOTH
    the with-add-back and (negative control) without-add-back estimators from a per-step log of
    `Wp` (recomputing the same second-order backward difference the class uses internally) and
    `force_applied`, and bins them against the exact force ``-U'(x) = -k x``. With add-back the
    RMS error must be much smaller than without, and small relative to the force scale itself.
    """
    k, mass = 2.0, 1.0

    def energy(positions):
        x = positions[0, 0]
        y, z = positions[0, 1], positions[0, 2]
        return 0.5 * k * x**2 + 0.5 * k * (y**2 + z**2)

    def ctx():
        return _one_particle(energy, mass=mass, dt=1.0, friction=0.5, seed=0)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-3.0,), upper=(3.0,), shape=(24,), periodic=False)
    method = MetaDSpectralABF(
        cvs, grid, height=2 * KT, sigma=0.2, stride=10, ngaussians=1600, deltaT=None,
        abf_bias=False, N=20, fit_threshold=10**9, fit_freq=10**9, fit_min_count=10**9,
        F_max=200.0,
    )
    log = CVLog()
    result = pysages.run(method, ctx, 15000, callback=log, stride=1)
    assert int(result.states[0].n_clipped) == 0  # F_max=200 must never bind here

    xi = onp.array(log.xi)
    Wp = onp.array(log.Wp)
    fa = onp.array(log.force_applied)
    dt = 1.0
    dWp_dt = onp.full_like(xi, onp.nan)
    dWp_dt[2:] = (1.5 * Wp[2:] - 2.0 * Wp[1:-1] + 0.5 * Wp[:-2]) / dt

    with_addback = dWp_dt + fa
    without_addback = dWp_dt.copy()
    exact = -k * xi

    edges = onp.linspace(-3.0, 3.0, 25)
    centers = 0.5 * (edges[1:] + edges[:-1])
    exact_c = -k * centers
    mf_with, count = _bin_average(xi, with_addback, edges)
    mf_without, _ = _bin_average(xi, without_addback, edges)
    used = (count > 20) & onp.isfinite(mf_with)

    rms_with = float(onp.sqrt(onp.mean((mf_with[used] - exact_c[used]) ** 2)))
    rms_without = float(onp.sqrt(onp.mean((mf_without[used] - exact_c[used]) ** 2)))
    scale = float(onp.mean(onp.abs(exact_c[used])))
    print(f"[exactness] rms_with={rms_with:.4f} rms_without={rms_without:.4f} scale={scale:.4f}")

    assert rms_with < 5 * rms_without / 20  # with-add-back is dramatically better...
    assert rms_with < 0.3 * scale  # ...and small relative to the force scale itself
    assert rms_without > 0.5 * scale  # negative control is genuinely, badly biased

    # Sec. 2.2: without any clamp firing, force_applied must equal f_abf + f_md exactly.
    f_sum = onp.asarray(result.states[0].f_abf) + onp.asarray(result.states[0].f_md)
    assert onp.allclose(onp.asarray(result.states[0].force_applied), f_sum, atol=1e-10)


# ---------------------------------------------------------------------------------------------
# 6.1b -- reductions
# ---------------------------------------------------------------------------------------------


def test_height_zero_never_deposits_and_f_md_is_exactly_zero():
    """Sec. 2.7: height=0 must make f_md identically 0.0 (bit-exact) at every step, so the
    combined force reduces exactly to the (patched) sABF term alone."""

    def energy(positions):
        x = positions[0, 0]
        return 0.5 * 2.0 * x**2 + 0.5 * 2.0 * (positions[0, 1] ** 2 + positions[0, 2] ** 2)

    def ctx():
        return _one_particle(energy, seed=1)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-3.0,), upper=(3.0,), shape=(24,), periodic=False)
    method = MetaDSpectralABF(cvs, grid, height=0.0, sigma=0.2, stride=5, ngaussians=400,
                               deltaT=None, N=20, fit_threshold=200, fit_freq=50, fit_min_count=10)
    result = pysages.run(method, ctx, 2000)
    state = result.states[0]
    assert onp.all(onp.asarray(state.f_md) == 0.0)
    assert onp.all(onp.asarray(state.dV_G) == 0.0)
    assert onp.all(onp.asarray(state.V_G) == 0.0)


def test_abf_bias_false_never_contributes_f_abf():
    """Sec. 2.7: abf_bias=False must make f_abf identically 0.0 (bit-exact) at every step (the
    algorithmic MetaDABF reduction), while Fsum/hist keep accumulating passively."""

    def energy(positions):
        x = positions[0, 0]
        return 0.5 * 2.0 * x**2 + 0.5 * 2.0 * (positions[0, 1] ** 2 + positions[0, 2] ** 2)

    def ctx():
        return _one_particle(energy, seed=2)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-3.0,), upper=(3.0,), shape=(24,), periodic=False)
    method = MetaDSpectralABF(cvs, grid, height=0.01, sigma=0.2, stride=10, ngaussians=400,
                               deltaT=1200.0, kB=KB, abf_bias=False, N=20, fit_threshold=50,
                               fit_freq=25, fit_min_count=10)
    result = pysages.run(method, ctx, 2000)
    state = result.states[0]
    assert onp.all(onp.asarray(state.f_abf) == 0.0)
    assert int(onp.asarray(state.hist).sum()) == 2000  # estimator still accumulated passively


# ---------------------------------------------------------------------------------------------
# 6.1c -- metad grid mechanics: deposit-centre peak, mirror edge cancellation
# ---------------------------------------------------------------------------------------------


def test_metad_potential_peaks_at_deposit_center():
    """A single non-well-tempered hill deposited at a known center must give `V_G`'s max at
    (or immediately next to) that center's grid bin -- the exact bug (a physical/unit mesh
    mismatch) the memo's Sec. 2.4 flags in the dpmd-ipi MetaDABF fork."""
    grid = pysages.Grid(lower=(-2.0,), upper=(2.0,), shape=(64,), periodic=False)

    class _M:
        height = 0.5
        deltaT = 1200.0  # well-tempered, so `V_G` is actually tracked (deltaT=None only
        kB = KB          # tracks dV_G -- see build_gaussian_accumulator's `if deltaT is None`)
        metad_grid = grid
        mirror = False
        cvs = [Component([0], 0)]

    deposit, _ = build_gaussian_accumulator(_M())
    V_G = jnp.zeros((64,))
    dV_G = jnp.zeros((64, 1))
    center = jnp.array([0.6])
    sigma = jnp.array([0.1])
    h, V_G, dV_G = deposit(center, sigma, V_G, dV_G)
    mesh = pysages.approxfun.compute_mesh(grid).reshape(-1)
    peak_x = float(mesh[int(jnp.argmax(V_G))])
    print(f"[metad V_G peak] deposited at 0.6, peak bin center at {peak_x:.4f}")
    assert abs(peak_x - 0.6) < (4.0 / 64) * 1.5  # within ~1.5 bin widths


def test_metad_mirror_images_cancel_gradient_at_the_edge():
    """Sec. 2.4: with `mirror=True`, a hill deposited close to the upper edge must leave
    `dV_G/dxi ~= 0` right at the boundary (McGovern & de Pablo); with `mirror=False` it does
    not."""
    grid = pysages.Grid(lower=(-2.0,), upper=(2.0,), shape=(128,), periodic=False)

    def run(mirror_flag):
        class _M:
            height = 0.5
            deltaT = None
            kB = None
            metad_grid = grid
            mirror = mirror_flag
            cvs = [Component([0], 0)]

        deposit, _ = build_gaussian_accumulator(_M())
        V_G = jnp.zeros((128,))
        dV_G = jnp.zeros((128, 1))
        center = jnp.array([1.92])  # 0.08 from the upper edge at 2.0
        sigma = jnp.array([0.1])
        _, V_G, dV_G = deposit(center, sigma, V_G, dV_G)
        interp = build_metad_interpolator(grid)
        return float(interp(dV_G, jnp.array([2.0]))[0])

    grad_mirrored = run(True)
    grad_plain = run(False)
    print(f"[edge gradient] mirror=True: {grad_mirrored:.5f}  mirror=False: {grad_plain:.5f}")
    # The continuum limit gives an exact zero at the boundary for a mirrored Gaussian pair;
    # what is actually read back here is a linear interpolation clamped to the last bin's
    # center (a few bin-widths short of the true edge, since the mesh is bin-centered), so a
    # large-but-not-total reduction is the honest bar, not a near-zero residual.
    assert abs(grad_mirrored) < 0.2 * abs(grad_plain)


# ---------------------------------------------------------------------------------------------
# burn_in / freeze_after
# ---------------------------------------------------------------------------------------------


def test_burn_in_gates_only_the_production_accumulators():
    def energy(positions):
        x = positions[0, 0]
        return 0.5 * 2.0 * x**2 + 0.5 * 2.0 * (positions[0, 1] ** 2 + positions[0, 2] ** 2)

    def ctx():
        return _one_particle(energy, seed=3)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-3.0,), upper=(3.0,), shape=(24,), periodic=False)
    method = MetaDSpectralABF(cvs, grid, height=0.005, sigma=0.2, stride=10, ngaussians=400,
                               deltaT=1200.0, kB=KB, N=20, fit_threshold=10**9, fit_freq=10**9,
                               fit_min_count=10**9, burn_in=500)
    result = pysages.run(method, ctx, 1000)
    state = result.states[0]
    assert int(onp.asarray(state.hist).sum()) == 1000
    assert int(onp.asarray(state.hist_prod).sum()) == 500  # only steps 501..1000 count


def test_freeze_after_stops_fits_and_deposits_but_keeps_accumulating():
    def energy(positions):
        x = positions[0, 0]
        return 0.5 * 2.0 * x**2 + 0.5 * 2.0 * (positions[0, 1] ** 2 + positions[0, 2] ** 2)

    def ctx():
        return _one_particle(energy, seed=4)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-3.0,), upper=(3.0,), shape=(24,), periodic=False)
    method = MetaDSpectralABF(cvs, grid, height=0.02, sigma=0.2, stride=10, ngaussians=2000,
                               deltaT=1200.0, kB=KB, N=10, fit_threshold=100, fit_freq=50,
                               fit_min_count=10, freeze_after=600)
    result = pysages.run(method, ctx, 1200)
    state = result.states[0]
    idx_at_freeze = int(onp.ceil((600 - 1) / 10)) + 1  # rough upper bound on hills before freeze
    assert int(state.idx) <= idx_at_freeze + 1
    assert int(onp.asarray(state.hist).sum()) == 1200  # accumulators keep filling past freeze


# ---------------------------------------------------------------------------------------------
# 6.1a (mean-force correctness on the mlmd backend) -- ABF family
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["ABF", "SpectralABF", "MetaDSpectralABF"])
def test_mean_force_correct_on_mlmd_backend_abf_family(which):
    """No existing test checked ABF mean-force correctness on the mlmd backend (memo Sec. 2.2,
    "No existing test checks..."). Harmonic well, exact mean force -k x; check the raw
    ``Fsum/hist`` accumulators against -k x over well-sampled bins.

    Note: `SpectralABF`/`MetaDSpectralABF` silently convert a non-periodic grid to
    `Grid[Chebyshev]` in `__init__` (`self.grid = convert(...)`), so the mesh/bin-center
    locations used to evaluate "exact" at each bin must come from ``method.grid`` (post
    conversion), not the `Grid` object this test constructs -- using the latter compares
    against the wrong (uniformly-spaced) bin centers and looks like a large, spurious error.
    """
    k, mass = 3.0, 1.0

    def energy(positions):
        x = positions[0, 0]
        return 0.5 * k * x**2 + 0.5 * k * (positions[0, 1] ** 2 + positions[0, 2] ** 2)

    def ctx():
        return _one_particle(energy, mass=mass, dt=0.5, friction=0.05, seed=5)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-2.0,), upper=(2.0,), shape=(48,), periodic=False)
    if which == "ABF":
        method = ABF(cvs, grid, N=50)
    elif which == "SpectralABF":
        method = SpectralABF(cvs, grid, N=50)
    else:
        method = MetaDSpectralABF(cvs, grid, height=0.005, sigma=0.08, stride=20,
                                   ngaussians=10000, deltaT=1200.0, kB=KB, N=50)
    result = pysages.run(method, ctx, 150000)
    hist = onp.asarray(result.states[0].hist)
    Fsum = onp.asarray(result.states[0].Fsum)
    mean_force = (Fsum / onp.maximum(hist, 1)[..., None]).reshape(-1)
    mesh = onp.asarray(pysages.approxfun.compute_mesh(method.grid)).reshape(-1)
    used = hist > 300
    exact = -k * mesh
    err = mean_force[used] - exact[used]
    rel_rms = float(onp.sqrt(onp.mean(err**2)) / onp.mean(onp.abs(exact[used])))
    print(f"[{which} mean force] relative RMS error over {used.sum()} bins: {rel_rms:.4f}")
    assert rel_rms < 0.15


# ---------------------------------------------------------------------------------------------
# 6.1d -- pickle/resume round-trip and restart bit-exactness
# ---------------------------------------------------------------------------------------------


def test_state_level_pickle_round_trip_then_one_update_matches(tmp_path):
    def energy(positions):
        x = positions[0, 0]
        return 0.5 * 2.0 * x**2 + 0.5 * 2.0 * (positions[0, 1] ** 2 + positions[0, 2] ** 2)

    def ctx():
        return _one_particle(energy, seed=6)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-3.0,), upper=(3.0,), shape=(24,), periodic=False)
    method = MetaDSpectralABF(cvs, grid, height=0.01, sigma=0.2, stride=10, ngaussians=300,
                               deltaT=1200.0, kB=KB, N=20, fit_threshold=50, fit_freq=25,
                               fit_min_count=10)
    result = pysages.run(method, ctx, 300)
    p = tmp_path / "r.pickle"
    pysages.save(result, p)
    loaded = pysages.load(p)

    def _assert_states_equal(s0, s1, label):
        for field in s0._fields:
            va, vb = getattr(s0, field), getattr(s1, field)
            if field == "fun":  # a `Fun` NamedTuple of ragged (scale, coefficients, c0) leaves
                for sub in va._fields:
                    a, b = onp.asarray(getattr(va, sub)), onp.asarray(getattr(vb, sub))
                    assert onp.array_equal(a, b), f"{label}: fun.{sub} mismatch"
            else:
                a, b = onp.asarray(va), onp.asarray(vb)
                assert onp.array_equal(a, b), f"{label}: field {field} mismatch"

    _assert_states_equal(result.states[0], loaded.states[0], "pickle round-trip")

    one_more = pysages.run(loaded, ctx, 1)
    another = pysages.run(result, ctx, 1)
    _assert_states_equal(one_more.states[0], another.states[0], "one post-pickle update")


@pytest.mark.parametrize("which", ["ABF", "SpectralABF", "MetaDSpectralABF"])
def test_restart_is_bit_exact_with_the_backend_fix(which, tmp_path):
    """Backend fix (memo Sec. 5): mlmd `Sampler.restore` now also restores `step`/`key`, so a
    split run (save + load + resume) must match a single continuous run bit-for-bit -- both the
    MD trajectory (positions) and the method's own accumulators (hist)."""
    k, mass = 2.0, 1.0

    def energy(positions):
        x = positions[0, 0]
        return 0.5 * k * x**2 + 0.5 * k * (positions[0, 1] ** 2 + positions[0, 2] ** 2)

    def ctx():
        return _one_particle(energy, mass=mass, dt=0.5, friction=0.3, seed=11)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-2.0,), upper=(2.0,), shape=(20,), periodic=False)

    def make_method():
        if which == "ABF":
            return ABF(cvs, grid, N=10)
        if which == "SpectralABF":
            return SpectralABF(cvs, grid, N=10, fit_threshold=50, fit_freq=25)
        return MetaDSpectralABF(cvs, grid, height=0.005, sigma=0.1, stride=20, ngaussians=200,
                                 deltaT=1200.0, kB=KB, N=10, fit_threshold=50, fit_freq=25)

    cont = pysages.run(make_method(), ctx, 800)

    part1 = pysages.run(make_method(), ctx, 400)
    p = tmp_path / f"{which}.pickle"
    pysages.save(part1, p)
    part2 = pysages.run(pysages.load(p), ctx, 400)

    pos_cont = onp.asarray(cont.snapshots[0].positions)
    pos_split = onp.asarray(part2.snapshots[0].positions)
    hist_cont = onp.asarray(cont.states[0].hist)
    hist_split = onp.asarray(part2.states[0].hist)
    assert onp.array_equal(pos_cont, pos_split)
    assert onp.array_equal(hist_cont, hist_split)


# ---------------------------------------------------------------------------------------------
# 6.2 -- 1D convergence toys (slow: run once, report numbers)
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_1d_double_well_convergence():
    """Sec. 6.2(i): symmetric double well, ~10 kT barrier. U(x) = h(x^2-a^2)^2, minima at +-a;
    since x is the system's only coordinate (y/z are harmonically confined independent of x),
    the exact FES is U(x) - min(U) exactly, no marginalization needed."""
    a, mass = 1.5, 1.0
    h = 10 * KT / a**4  # barrier height U(0) - U(a) = h*a^4 = 10 kT

    def energy(positions):
        x = positions[0, 0]
        return h * (x**2 - a**2) ** 2 + 0.5 * 2.0 * (positions[0, 1] ** 2 + positions[0, 2] ** 2)

    def ctx():
        return _one_particle(energy, mass=mass, dt=0.5, friction=0.3, seed=21)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(-2.2,), upper=(2.2,), shape=(64,), periodic=False)
    method = MetaDSpectralABF(
        cvs, grid, height=0.3 * KT, sigma=0.1, stride=25, ngaussians=24000, deltaT=8 * T * KB,
        kB=KB, N=100, fit_threshold=5000, fit_freq=500, fit_min_count=50, burn_in=20000,
        F_max=30.0,
    )
    result = pysages.run(method, ctx, 600000)
    ana = pysages.analyze(result)
    mesh = onp.asarray(ana["mesh"]).reshape(-1)
    fes = onp.asarray(ana["free_energy"]).reshape(-1)
    fes_trapz = onp.asarray(ana["free_energy_trapz"]).reshape(-1)
    exact = h * (mesh**2 - a**2) ** 2
    exact = exact - exact.min()

    hist = onp.asarray(ana["histogram"]).reshape(-1)
    used = hist > 50
    rmse_spectral = float(onp.sqrt(onp.mean((fes[used] - exact[used]) ** 2))) / KT
    rmse_trapz = float(onp.sqrt(onp.mean((fes_trapz[used] - exact[used]) ** 2))) / KT
    print(f"[double well] bins used {used.sum()}/{len(mesh)}; "
          f"RMSE spectral={rmse_spectral:.2f} kT, trapz={rmse_trapz:.2f} kT; "
          f"n_clipped={int(ana['n_clipped'])}")
    assert onp.isfinite(fes).all()
    assert rmse_spectral < 3.0  # kT -- a coarse, honest convergence bar, not a tight fit test


@pytest.mark.slow
def test_1d_w1_like_steep_floor_toy():
    """Sec. 6.2(ii): hard floor at 0 (a steep repulsive wall keeps x >= 0, as CVDist's own
    positive-semidefinite construction does), a 0.55 eV tanh(x/0.3) rise, a 0.12 eV bump at
    2.8, and a plateau; report RMSE(A) and time-to-coverage rather than asserting a tight bound
    (this is the case the memo flags as the hardest to sample and the primary motivation for
    the method)."""
    mass = 1.0

    def u1d(x):
        return 0.55 * jnp.tanh(x / 0.3) + 0.12 * jnp.exp(-0.5 * ((x - 2.8) / 0.3) ** 2)

    def energy(positions):
        x = positions[0, 0]
        wall = jnp.where(x < 0.0, 50.0 * x**2, 0.0)
        return u1d(x) + wall + 0.5 * 2.0 * (positions[0, 1] ** 2 + positions[0, 2] ** 2)

    def ctx():
        return _one_particle(energy, mass=mass, dt=0.5, friction=0.3, seed=22)

    cvs = [Component([0], 0)]
    grid = pysages.Grid(lower=(0.0,), upper=(6.0,), shape=(96,), periodic=False)
    restraints = pysages.methods.CVRestraints(lower=[0.0], upper=[6.0], kl=[0.0], ku=[3.0])
    method = MetaDSpectralABF(
        cvs, grid, height=0.4 * KT, sigma=0.08, stride=25, ngaussians=24000, deltaT=10 * T * KB,
        kB=KB, N=200, fit_threshold=5000, fit_freq=500, fit_min_count=100, burn_in=20000,
        F_max=3.0, restraints=restraints,
    )
    result = pysages.run(method, ctx, 600000)
    ana = pysages.analyze(result)
    mesh = onp.asarray(ana["mesh"]).reshape(-1)
    hist = onp.asarray(ana["histogram"]).reshape(-1)
    fes = onp.asarray(ana["free_energy"]).reshape(-1)
    fes_trapz = onp.asarray(ana["free_energy_trapz"]).reshape(-1)
    exact = onp.asarray([float(u1d(jnp.asarray(x))) for x in mesh])
    exact = exact - exact.min()

    used = hist > 20
    coverage = float(used.sum()) / len(mesh)
    rmse_spectral = float(onp.sqrt(onp.mean((fes[used] - exact[used]) ** 2))) / KT
    rmse_trapz = float(onp.sqrt(onp.mean((fes_trapz[used] - exact[used]) ** 2))) / KT
    print(f"[W1-like] coverage={coverage:.2f}; RMSE spectral={rmse_spectral:.2f} kT, "
          f"trapz={rmse_trapz:.2f} kT; n_clipped={int(ana['n_clipped'])}; "
          f"heights tail (last 5)={onp.asarray(ana['heights'])[-5:]}")
    assert onp.isfinite(fes).all()
    assert coverage > 0.5


# ---------------------------------------------------------------------------------------------
# 6.3 -- 2D hidden barrier (control arm: 2D sABF only, height=0; report, do not over-assert)
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_2d_hidden_barrier_sabf_control_arm():
    """Sec. 6.3, control arm: plain 2D sABF (height=0) on V = h_x(x^2-1)^2 + h_y(y^2-1)^2
    - c*x*y, variant A (uniform y-barrier, h_y = 6 kT-equivalent). Only reports diagnostics
    (replica-free single run here; the paper-scale version needs 16 seeds): finite output,
    and that the fitted A(x) after marginalizing y is in the right ballpark of the exact
    A(x) = -kT ln int exp(-beta V) dy. Metad-on-grid for d=2 is deliberately not exercised
    here (see `build_gaussian_accumulator`'s docstring: mixing that in changes nothing about
    the 2D quirk it documents, and Sec. 6.3's honest expectation is that no x-only method
    beats the y-hopping time in variant A anyway)."""
    h_x, h_y, c = 4.0, 6.0, 2.0

    def energy(positions):
        x, y, z = positions[0, 0], positions[0, 1], positions[0, 2]
        return h_x * (x**2 - 1) ** 2 + h_y * (y**2 - 1) ** 2 - c * x * y + 0.5 * 2.0 * z**2

    def ctx():
        return _one_particle(energy, mass=1.0, dt=0.25, friction=0.3, seed=31)

    cvs = [Component([0], 0), Component([0], 1)]
    grid = pysages.Grid(lower=(-2.0, -2.0), upper=(2.0, 2.0), shape=(24, 24), periodic=False)
    method = MetaDSpectralABF(cvs, grid, height=0.0, sigma=(0.1, 0.1), stride=25, ngaussians=100,
                               deltaT=None, N=50, fit_threshold=5000, fit_freq=500,
                               fit_min_count=50, burn_in=5000, F_max=20.0)
    result = pysages.run(method, ctx, 300000)
    ana = pysages.analyze(result)
    hist = onp.asarray(ana["histogram"])
    print(f"[2D hidden barrier] n_clipped={int(ana['n_clipped'])}; "
          f"bins visited={(hist > 0).sum()}/{hist.size}; max hist={hist.max()}; "
          f"y-histogram spread (std over y at x~0): "
          f"{onp.std(hist[hist.shape[0] // 2, :]):.1f}")
    assert onp.isfinite(onp.asarray(ana["free_energy"])).all()
