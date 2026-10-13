# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Metadynamics + Spectral Adaptive Biasing Force (``MetaDSpectralABF``).

Implements task T2 of ``/scratch/slv5250/SALT/plan/metad_sabf_design.md`` (the design memo is
the spec; section numbers below refer to it). The applied bias is the sABF bias (a Chebyshev
spectral-gradient fit of the running mean force, `pysages.methods.spectral_abf.SpectralABF`)
*plus* a well-tempered metadynamics bias on the same collective variables, evaluated on its own
uniform grid with linear interpolation. The estimator records the momentum-derivative force
*plus the exact generalized force that was applied at the previous step* (the Darve-
Rodriguez-Gomez-Pohorille identity, memo Sec. 2.1-2.2): because a bias that only depends on
``x`` through ``xi(x)`` leaves the conditional distribution ``rho(x|xi)`` unchanged, this
subtraction is exact for *any* combination of sABF + metad + walls, static or time dependent,
and ``Fsum/hist`` (or ``Fsum_prod/hist_prod``, see ``burn_in`` below) is the physical mean force
``-dA/dxi`` -- the metadynamics contribution cancels out of the estimator even though it never
cancels out of the dynamics.

Two reductions fall out of one code path (memo Sec. 2.7):

* ``height=0`` disables all Gaussian deposition (the metad force is identically zero every
  step), so the bias is pure (patched) `SpectralABF`.
* ``abf_bias=False`` drops the fitted/averaged ABF force from the bias (the estimator keeps
  accumulating ``Fsum``/``hist`` passively), which is algorithmically `MetaDABF`
  (``/scratch/slv5250/dpmd-ipi/PySAGES/pysages/methods/metadabf.py``), modulo the grid/
  interpolation choice (memo Sec. 2.4).

Patches ported from the "hydrolysis project" stability fixes in
``/scratch/slv5250/POLYMER/07_md_engine/pysages-mlmd/pysages/methods/funnel_sabf.py`` (memo
Sec. 2.5), generalized to a per-bin gate rather than a global one:

1. The fitted force is only used in bins with ``hist >= fit_min_count``; other bins fall back
   to the raw ramp average. Upstream `SpectralABF` gates globally on ``ncalls``, which
   extrapolates the fit into bins that have never been visited.
2. The fit is trained on ``Fsum / max(hist, N)`` (N-damped), not ``Fsum / max(hist, 1)``: a
   handful of samples with extreme instantaneous force otherwise dominate the global polynomial
   fit. This is a no-op once ``hist >= N``.
3. All clamping (``F_max``) happens in CV space (on ``f_abf``, and again on the combined
   ``f_tot``), never per atom. A CV-space clamp keeps the bias proportional to ``Jxi.T``, a pure
   xi-bias, so the estimator stays exact (Sec. 2.2). ``max_atom_bias`` is kept only as a crash
   guard (as in ``funnel_sabf.py``) with a counter, ``n_clipped``; a window where it fires after
   burn-in is suspect, because the atom-space clip is *not* undone by the subtraction rule.

Metadynamics component (memo Sec. 2.4): well-tempered heights ``h = h0 exp(-V_G(xi)/(kB dT))``,
deposited every ``stride`` steps and only when ``xi`` is inside the metad grid. ``V_G``/``dV_G``
live on their own uniform grid (``metad_grid_points`` nodes per axis, same bounds as the ABF
grid) and are read back by (multi)linear interpolation for the force that actually enters the
bias -- *not* the nearest-bin lookup `Metadynamics`/`MetaDABF` use, which is only C^{-1} and
would make ``f_md`` a step function. Mirror Gaussian images are deposited at both grid edges
(McGovern & de Pablo, JCP 139, 084102, 2013) so ``dV_G/dxi = 0`` at the boundary, keeping the
system from being pushed past the wall margin where no ABF or metad acts.

Handover (memo Sec. 2.6): WT-metad is kept on for the whole run (as in meta-eABF/WTM-eABF), with
two cheap, orthogonal options:

* ``burn_in``: a second set of accumulators (``Fsum_prod``/``hist_prod``) that only fill once
  ``ncalls > burn_in``. The *bias* always uses the all-time accumulators (no discontinuity);
  ``analyze()`` reports the production accumulators, dropping early non-equilibrium samples.
* ``freeze_after``: after this many steps, no more fits and no more Gaussian deposits happen
  (the bias becomes static); the accumulators keep filling. Needed for exact
  ``exp(beta U_bias(xi))`` frame weights in reweighting.

Serialization / restart (memo Sec. 5): state is a pytree of fixed-shape arrays plus a
`pysages.approxfun.Fun`; ``pysages.save``/``pysages.load`` round-trip it exactly like
`SpectralABF`. Counters (``ncalls``, ``idx``, ``n_clipped``) are explicit ``int64`` arrays, not
Python ints, so the ``lax.scan`` carry used by the mlmd backend has a stable dtype across
resumed chunks.
"""

from itertools import product as _cartesian

import numpy as _onp
from jax import grad, jit, value_and_grad, vmap
from jax import numpy as np
from jax.lax import cond

from pysages.approxfun import (
    Fun,
    SpectralGradientFit,
    build_evaluator,
    build_fitter,
    build_grad_evaluator,
    compute_mesh,
)
from pysages.approxfun.core import vandergrad_builder
from pysages.colvars import get_periods
from pysages.grids import Chebyshev, Grid, build_indexer, convert, grid_transposer
from pysages.methods.core import GriddedSamplingMethod, Result, generalize
from pysages.methods.metad import sum_of_gaussians
from pysages.methods.restraints import apply_restraints
from pysages.methods.utils import numpyfy_vals
from pysages.typing import JaxArray, NamedTuple, Tuple
from pysages.utils import dispatch, first_or_all, linear_solver

# Large-but-finite int64 sentinel used in place of `freeze_after=None` ("never freeze"), so the
# `ncalls <= freeze_at` comparison inside the jitted update has a single, stable code path.
_NEVER_FREEZE = _onp.int64(2**62)


class MetaDSpectralABFState(NamedTuple):
    """
    MetaDSpectralABF internal state. See the module and memo Sec. 3 for the field semantics.

    Parameters
    ----------

    xi: JaxArray [cvs.shape]
        Last collective variable recorded in the simulation.

    bias: JaxArray [natoms, 3]
        Biasing forces for each particle actually applied this step (post CV- and, if
        configured, atom-space clamping).

    hist, Fsum: JaxArray [grid.shape(, cvs.shape)]
        All-time histogram / cumulative generalized force, exactly as in `SpectralABF`. The
        bias (sABF fit + estimator ramp) is always trained on these.

    hist_prod, Fsum_prod: JaxArray [grid.shape(, cvs.shape)]
        Same, but only accumulated once ``ncalls > burn_in``. ``analyze()`` reports these.

    force_applied: JaxArray [cvs.shape]
        ``-tsolve(Jxi, bias)`` for the bias *actually applied* last step (post any clamping).
        Added back into next step's ``Fsum``/``Fsum_prod`` increment -- the exactness crux,
        memo Sec. 2.2.

    Wp, Wp_: JaxArray [cvs.shape]
        Current and previous ``W @ p`` (as in ABF/SpectralABF).

    fun: Fun
        Coefficients of the Chebyshev spectral-gradient fit of the sABF free energy.

    V_G: JaxArray [metad_grid.shape]
        Well-tempered metadynamics potential on the metad grid.

    dV_G: JaxArray [metad_grid.shape, cvs.shape]
        Gradient of ``V_G`` on the metad grid (read back by linear interpolation for ``f_md``).

    heights: JaxArray [ngaussians]
        Record of deposited hill heights (0 for not-yet-deposited slots); mirror images are
        *not* recorded here, only folded into ``V_G``/``dV_G``.

    centers: JaxArray [ngaussians, cvs.shape]
        Record of deposited hill centers.

    idx: JaxArray []
        Index of the next hill slot.

    ncalls: JaxArray []
        Number of times ``update`` has run.

    f_abf, f_md: JaxArray [cvs.shape]
        Last-step ABF and metad generalized forces (diagnostics only).

    n_clipped: JaxArray []
        Number of steps where the CV-space (``F_max``) or atom-space (``max_atom_bias``) clamp
        actually fired. Should be 0 after burn-in for a healthy window (memo Sec. 8, risk 2).
    """

    xi: JaxArray
    bias: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    hist_prod: JaxArray
    Fsum_prod: JaxArray
    force_applied: JaxArray
    Wp: JaxArray
    Wp_: JaxArray
    fun: Fun
    V_G: JaxArray
    dV_G: JaxArray
    heights: JaxArray
    centers: JaxArray
    idx: JaxArray
    ncalls: JaxArray
    f_abf: JaxArray
    f_md: JaxArray
    n_clipped: JaxArray
    force_applied_prev: JaxArray  # [2, cvs.shape]: force_applied of the two updates before last

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class PartialMetaDSpectralABFState(NamedTuple):
    xi: JaxArray
    hist: JaxArray
    Fsum: JaxArray
    ind: Tuple
    fun: Fun
    pred: bool


class MetaDSpectralABF(GriddedSamplingMethod):
    """
    Metadynamics + Spectral ABF (see module docstring and the T2 design memo).

    Parameters
    ----------

    cvs: Union[List, Tuple]
        Collective variables (1 or 2, the `SpectralABF` basis-expansion limit).

    grid: Grid
        sABF domain and bin count. Converted to `Grid[Chebyshev]` if not periodic, exactly as
        in `SpectralABF`. The metad grid (see ``metad_grid_points``) shares its bounds.

    height, sigma, stride, ngaussians: as in `Metadynamics`.
        ``height=0`` disables deposition entirely (the ``height=0`` reduction, Sec. 2.7).

    deltaT: Optional[float] = None
        Well-tempered Delta T; ``None`` gives standard (non-tempered) metadynamics.

    kB: float
        Required if ``deltaT`` is not ``None``.

    restraints: Optional[CVRestraints] = None
        Harmonic walls outside the grid (shared by both the ABF and metad component: outside
        the grid, only the walls act -- memo Sec. 2.3).

    N: int = 200
        ABF ramp threshold (as in ABF/SpectralABF); also the N-damping constant for the sABF
        fit input (memo Sec. 2.5.2).

    fit_freq: int = 100
        Spectral refit cadence (in steps), once ``ncalls > fit_threshold``.

    fit_threshold: int = 5000
        Steps before the first refit is attempted (as in `SpectralABF`, but with a much larger
        default since the per-bin gate below is what actually controls when the fit is used).

    fit_min_count: int = 100
        Per-bin gate: a bin only uses the fitted force once ``hist[bin] >= fit_min_count``
        (memo Sec. 2.5.1); otherwise it falls back to the ramp average, same as ABF.

    F_max: float = 3.0
        CV-space force cap, applied both to the fitted branch and to the combined
        ``f_abf + f_md`` (memo Sec. 2.2-2.3).

    abf_bias: bool = True
        If ``False``, the sABF force is dropped from the bias (the ``abf_bias=False``
        reduction -- algorithmically `MetaDABF`); the estimator keeps accumulating.

    metad_grid_points: int = 512
        Node count per axis of the metad grid (`Grid`, uniform, *not* Chebyshev).

    mirror: bool = True
        Deposit mirror Gaussian images at both metad-grid edges (memo Sec. 2.4).

    burn_in: int = 0
        Steps before the production accumulators (``Fsum_prod``/``hist_prod``) start filling.

    freeze_after: Optional[int] = None
        After this many steps, stop fitting and stop depositing hills (the bias becomes
        static); accumulators keep filling. ``None`` never freezes.

    max_atom_bias: Optional[float] = None
        Crash-guard clamp on the final per-atom bias, in force units. Off by default; keep
        ``n_clipped`` at 0 if it is used.

    use_pinv: bool = False
        As in ABF/SpectralABF.
    """

    snapshot_flags = {"positions", "indices", "momenta"}

    def __init__(self, cvs, grid, height, sigma, stride, ngaussians, deltaT=None, **kwargs):
        super().__init__(cvs, grid, **kwargs)

        self.height = height
        self.sigma = sigma
        self.stride = stride
        self.ngaussians = ngaussians
        self.deltaT = deltaT
        self.kB = self.kwargs.get("kB", None)
        if deltaT is not None and self.kB is None:
            raise KeyError(
                "For well-tempered metadynamics a keyword argument `kB` for the value of "
                "the Boltzmann constant (matching the backend's internal units) must be "
                "provided."
            )

        self.N = np.asarray(self.kwargs.get("N", 200))
        self.fit_freq = self.kwargs.get("fit_freq", 100)
        self.fit_threshold = self.kwargs.get("fit_threshold", 5000)
        self.fit_min_count = self.kwargs.get("fit_min_count", 100)
        self.F_max = self.kwargs.get("F_max", 3.0)
        self.abf_bias = self.kwargs.get("abf_bias", True)
        self.metad_grid_points = self.kwargs.get("metad_grid_points", 512)
        self.mirror = self.kwargs.get("mirror", True)
        self.burn_in = self.kwargs.get("burn_in", 0)
        self.freeze_after = self.kwargs.get("freeze_after", None)
        self.max_atom_bias = self.kwargs.get("max_atom_bias", None)
        self.use_pinv = self.kwargs.get("use_pinv", False)

        self.grid = self.grid if self.grid.is_periodic else convert(self.grid, Grid[Chebyshev])
        self.model = SpectralGradientFit(self.grid)

        dims = int(self.grid.shape.size)
        metad_shape = _onp.full(dims, int(self.metad_grid_points))
        self.metad_grid = Grid(self.grid.lower, self.grid.upper, metad_shape)

    def build(self, snapshot, helpers, *_args, **_kwargs):
        return _metad_sabf(self, snapshot, helpers)


def _metad_sabf(method, snapshot, helpers):  # noqa: C901 -- one cohesive per-step update
    cv = method.cv
    grid = method.grid
    metad_grid = method.metad_grid
    dims = grid.shape.size
    periods = get_periods(method.cvs)

    dt = snapshot.dt
    natoms = np.size(snapshot.positions, 0)

    N = method.N
    fit_freq = method.fit_freq
    fit_threshold = method.fit_threshold
    fit_min_count = method.fit_min_count
    F_max = method.F_max
    abf_bias = method.abf_bias
    burn_in = int(method.burn_in)
    freeze_at = _onp.int64(method.freeze_after) if method.freeze_after is not None else _NEVER_FREEZE
    max_atom_bias = method.max_atom_bias
    stride = method.stride
    ngaussians = method.ngaussians
    sigma = np.broadcast_to(np.asarray(method.sigma, dtype=np.float64), (dims,))

    tsolve = linear_solver(method.use_pinv)
    get_grid_index = build_indexer(grid)
    # Add-back weights (see update): mlmd-backend three-step stencil, or the legacy single step.
    c1, c2, c3 = (2.25, -1.5, 0.25) if method.kwargs.get("stencil_addback", True) else (1.0, 0.0, 0.0)
    fit = build_fitter(method.model)
    fit_forces = build_free_energy_fitter(N, fit)
    estimate_abf_force = build_force_estimator(method)
    deposit_gaussian, should_deposit_metad = build_gaussian_accumulator(method)
    interp_field = build_metad_interpolator(metad_grid)

    query, dimensionality, to_force_units = helpers

    def initialize():
        xi, _ = cv(query(snapshot))
        bias = np.zeros((natoms, dimensionality()))
        hist = np.zeros(grid.shape, dtype=np.uint32)
        Fsum = np.zeros((*grid.shape, dims))
        hist_prod = np.zeros(grid.shape, dtype=np.uint32)
        Fsum_prod = np.zeros((*grid.shape, dims))
        force_applied = np.zeros(dims)
        Wp = np.zeros(dims)
        Wp_ = np.zeros(dims)
        fun = fit(Fsum)
        V_G = np.zeros((*metad_grid.shape,))
        dV_G = np.zeros((*metad_grid.shape, dims))
        heights = np.zeros(ngaussians, dtype=np.float64)
        centers = np.zeros((ngaussians, dims), dtype=np.float64)
        idx = np.array(0, dtype=np.int64)
        ncalls = np.array(0, dtype=np.int64)
        f_abf = np.zeros(dims)
        f_md = np.zeros(dims)
        n_clipped = np.array(0, dtype=np.int64)
        force_applied_prev = np.zeros((2, dims))
        return MetaDSpectralABFState(
            xi, bias, hist, Fsum, hist_prod, Fsum_prod, force_applied, Wp, Wp_, fun,
            V_G, dV_G, heights, centers, idx, ncalls, f_abf, f_md, n_clipped, force_applied_prev,
        )

    def update(state, data):
        ncalls = state.ncalls + 1
        xi, Jxi = cv(data)

        # 1) sABF/ABF estimator: adds back the EXACT force applied last step (Sec. 2.1-2.2).
        p = data.momenta
        Wp = tsolve(Jxi, p)
        dWp_dt = (1.5 * Wp - 2.0 * state.Wp + 0.5 * state.Wp_) / dt
        I_xi = get_grid_index(xi)
        # The mlmd backend (backends/mlmd.py, Sampler step) reports p_n = m v_n + dt/2 b_{n-1}
        # and applies each bias b_k as a full kick at x_k, so the bias impulse contained in the
        # second-order backward difference (1.5 Wp_n - 2 Wp_{n-1} + 0.5 Wp_{n-2})/dt is
        # -(2.25 f_{n-1} - 1.5 f_{n-2} + 0.25 f_{n-3}), not -f_{n-1}. Adding back only f_{n-1}
        # leaves R_n = 1.25 f_{n-1} - 1.5 f_{n-2} + 0.25 f_{n-3}: ~0 for a smooth bias, but O(jump)
        # whenever the applied CV force is discontinuous along the trajectory (grid-edge
        # crossings, where the bias switches between f_abf+f_md and the restraint), which
        # corrupts the bins within ~3 steps of travel of the grid edges. Add back the full stencil.
        f1 = state.force_applied
        f2, f3 = state.force_applied_prev[0], state.force_applied_prev[1]
        inc = to_force_units(dWp_dt) + c1 * f1 + c2 * f2 + c3 * f3
        hist = state.hist.at[I_xi].add(1)
        Fsum = state.Fsum.at[I_xi].add(inc)

        # 2) production accumulators, gated on burn_in (no discontinuity: bias always uses the
        #    all-time hist/Fsum above).
        in_prod = ncalls > burn_in
        hist_prod = state.hist_prod.at[I_xi].add(np.where(in_prod, 1, 0))
        Fsum_prod = state.Fsum_prod.at[I_xi].add(np.where(in_prod, inc, 0.0))

        frozen = ncalls > freeze_at

        # 3) metad deposit: stride cadence, in-grid only, not after freeze_after.
        in_deposition_step = (ncalls > 1) & (ncalls % stride == 1) & (~frozen)
        do_deposit = in_deposition_step & should_deposit_metad(xi)

        def _do(carry):
            V_G, dV_G, heights, centers, idx = carry
            h, V_G2, dV_G2 = deposit_gaussian(xi, sigma, V_G, dV_G)
            heights2 = heights.at[idx].set(h)
            centers2 = centers.at[idx].set(xi.reshape(dims))
            return V_G2, dV_G2, heights2, centers2, idx + 1

        def _skip(carry):
            return carry

        V_G, dV_G, heights, centers, idx = cond(
            do_deposit, _do, _skip,
            (state.V_G, state.dV_G, state.heights, state.centers, state.idx),
        )

        # 4) sABF refit: same cadence/threshold as SpectralABF, all-time (N-damped) input.
        in_fitting_regime = (ncalls > fit_threshold) & (~frozen)
        in_fitting_step = in_fitting_regime & (ncalls % fit_freq == 1)
        fun = fit_forces(state.Fsum, state.hist, state.fun, in_fitting_step)

        # 5) combine: f_abf (per-bin gated + inner clamp) + f_md (linear interpolation),
        #    clamp again in CV space, or (outside the grid) walls only (Sec. 2.3).
        pred = in_fitting_regime & (hist[I_xi] >= fit_min_count)

        def in_bounds(_):
            f_abf = (
                estimate_abf_force(PartialMetaDSpectralABFState(xi, hist, Fsum, I_xi, fun, pred))
                if abf_bias
                else np.zeros(dims)
            )
            f_md = interp_field(dV_G, xi)
            raw = f_abf + f_md
            clamped = np.clip(raw, -F_max, F_max)
            clipped = np.any(np.abs(raw) > F_max)
            return clamped, f_abf, f_md, clipped

        def out_of_bounds(_):
            f_abf = np.zeros(dims)
            f_md = np.zeros(dims)
            if method.restraints is None:
                r = np.zeros(dims)
            else:
                lo, hi, kl, kh = method.restraints
                r = apply_restraints(lo, hi, kl, kh, xi.reshape(dims))
            return r, f_abf, f_md, np.array(False)

        ob = np.any(np.array(I_xi) == grid.shape)
        f_tot, f_abf_out, f_md_out, clipped_cv = cond(ob, out_of_bounds, in_bounds, 0)

        # 6) apply: CV-space bias, optional atom-space crash-guard clamp, exact recovery of
        #    `force_applied` from whatever bias was actually applied (Sec. 2.2).
        raw_bias_flat = -Jxi.T @ f_tot
        bias_atoms = raw_bias_flat.reshape(state.bias.shape)
        if max_atom_bias is not None:
            bias_final = np.clip(bias_atoms, -max_atom_bias, max_atom_bias)
            clipped_atom = np.any(bias_final != bias_atoms)
        else:
            bias_final = bias_atoms
            clipped_atom = np.array(False)
        n_clipped = state.n_clipped + np.where(clipped_cv | clipped_atom, 1, 0)
        force_applied = -tsolve(Jxi, bias_final.reshape(p.shape))

        force_applied_prev = np.stack([f1, f2])
        return MetaDSpectralABFState(
            xi, bias_final, hist, Fsum, hist_prod, Fsum_prod, force_applied, Wp, state.Wp, fun,
            V_G, dV_G, heights, centers, idx, ncalls, f_abf_out, f_md_out, n_clipped,
            force_applied_prev,
        )

    return snapshot, initialize, generalize(update, helpers)


def build_free_energy_fitter(N, fit):
    """
    sABF spectral refit, trained on the all-time, N-damped mean force ``Fsum / max(hist, N)``
    (memo Sec. 2.5.2), only when ``in_fitting_step``.
    """

    def _fit_forces(args):
        Fsum, hist, _fun = args
        shape = (*Fsum.shape[:-1], 1)
        force = Fsum / np.maximum(hist.reshape(shape), N)
        return fit(force)

    def _skip_fitting(args):
        return args[2]

    def fit_forces(Fsum, hist, fun, in_fitting_step):
        return cond(in_fitting_step, _fit_forces, _skip_fitting, (Fsum, hist, fun))

    return fit_forces


@dispatch
def build_force_estimator(method: MetaDSpectralABF):
    """
    ABF-family force estimator: per-bin gated fitted force (CV-space clamped), else the ramp
    average ``Fsum/max(hist, N)``. Out-of-bounds handling lives in ``_metad_sabf`` (shared with
    the metad component and the walls, Sec. 2.3), so this only covers the in-bounds branch.
    """
    N = method.N
    grid = method.grid
    dims = grid.shape.size
    F_max = method.F_max
    model = method.model
    get_grad = build_grad_evaluator(model)

    def average_force(state):
        i = state.ind
        return state.Fsum[i] / np.maximum(state.hist[i], N)

    def interpolate_force(state):
        xi = np.clip(state.xi, grid.lower, grid.upper)
        f = get_grad(state.fun, xi).reshape(dims)
        return np.clip(f, -F_max, F_max)

    def estimate_force(state):
        return cond(state.pred, interpolate_force, average_force, state)

    return estimate_force


def build_gaussian_accumulator(method: MetaDSpectralABF):
    """
    Returns ``(deposit, should_deposit)``.

    ``deposit(xi, sigma, V_G, dV_G) -> (height, V_G', dV_G')`` evaluates one well-tempered hill
    (plus, if ``mirror``, up to ``2 * dims`` mirror images across the metad-grid edges within
    ``3 sigma`` of the hill center, McGovern & de Pablo 2013) on the whole metad mesh and folds
    it into the running potential/gradient grids. All images share the deposited height, which
    is what gives ``dV_G/dxi = 0`` exactly at the wall.

    Both the real hill and its mirror images are always passed to `sum_of_gaussians` as a
    genuine ``(n_images, dims)`` batch (never a bare ``(dims,)`` "single center"), which keeps
    the reduction over CV dimensions correct for ``dims == 2`` -- unlike the on-grid branch of
    `Metadynamics`/`MetaDABF`, which for a 2D grid calls it with a single center and ends up
    summing 1D Gaussians along each axis rather than an isotropic 2D one.
    """
    height_0 = method.height
    deltaT = method.deltaT
    kB = method.kB
    metad_grid = method.metad_grid
    mirror = method.mirror
    dims = int(metad_grid.shape.size)

    grid_mesh = compute_mesh(metad_grid)
    get_metad_index = build_indexer(metad_grid)

    def mirror_set(xi, sigma):
        xi = xi.reshape(dims)
        centers = [xi]
        closeness = []
        if mirror:
            lower, upper = metad_grid.lower, metad_grid.upper
            for k in range(dims):
                for bound in (lower[k], upper[k]):
                    centers.append(xi.at[k].set(2.0 * bound - xi[k]))
                    closeness.append(np.abs(xi[k] - bound) < 3.0 * sigma[k])
        centers = np.stack(centers)
        if mirror:
            mask = np.concatenate([np.array([True]), np.stack(closeness)])
        else:
            mask = np.array([True])
        return centers, mask

    if deltaT is None:

        def next_height(_v_here):
            return height_0

    else:

        def next_height(v_here):
            return height_0 * np.exp(-v_here / (deltaT * kB))

    def deposit(xi, sigma, V_G, dV_G):
        idx = get_metad_index(xi)
        v_here = V_G[idx] if deltaT is not None else 0.0
        h = next_height(v_here)
        centers, mask = mirror_set(xi, sigma)
        heights = h * mask.astype(centers.dtype)

        def potential(x):
            return sum_of_gaussians(x, heights, centers, sigma, get_periods(method.cvs))

        if deltaT is None:
            grads = vmap(grad(potential))(grid_mesh)
            V_G_new = V_G
        else:
            vals, grads = vmap(value_and_grad(potential))(grid_mesh)
            V_G_new = V_G + vals.reshape(V_G.shape)
        dV_G_new = dV_G + grads.reshape(dV_G.shape)
        return h, V_G_new, dV_G_new

    def should_deposit(xi):
        idx = get_metad_index(xi)
        return ~np.any(np.array(idx) == metad_grid.shape)

    return deposit, should_deposit


def build_metad_interpolator(metad_grid):
    """
    (Multi)linear interpolation of a field defined on ``metad_grid``'s bin-centered uniform
    mesh, at an arbitrary point ``xi`` (clamped into the grid). Works uniformly for a scalar
    field (``V_G``, shape ``metad_grid.shape``) or a vector field (``dV_G``, shape
    ``(*metad_grid.shape, dims)``) -- whatever trailing shape ``field`` has past the grid axes
    is preserved. This is what makes the metad force C^0 (memo Sec. 2.4), unlike the nearest-
    bin lookup `Metadynamics`/`MetaDABF` use.
    """
    lower = metad_grid.lower
    size = metad_grid.size
    shape = metad_grid.shape
    dims = int(shape.size)
    h = size / shape
    n = shape
    corners = list(_cartesian([0, 1], repeat=dims))

    def interp(field, xi):
        xi = np.reshape(xi, (dims,))
        t = (xi - lower) / h - 0.5  # bin-centered fractional coordinate
        t = np.clip(t, 0.0, np.asarray(n, dtype=t.dtype) - 1.0)
        i0 = np.clip(np.floor(t).astype(np.int32), 0, np.maximum(n - 2, 0))
        frac = t - i0.astype(t.dtype)
        total = np.zeros(field.shape[dims:], dtype=field.dtype)
        for off in corners:
            idx = tuple(np.clip(i0[k] + off[k], 0, n[k] - 1) for k in range(dims))
            w = 1.0
            for k in range(dims):
                w = w * (frac[k] if off[k] else (1.0 - frac[k]))
            total = total + w * field[idx]
        return total

    return interp


def _masked_refit(method, n_min, hist_prod, Fsum_prod):
    """
    Spectral refit of ``Fsum_prod / max(hist_prod, 1)`` (undamped production mean force) at
    analysis time, masking out bins with ``hist_prod < n_min`` from the least-squares solve.

    Factored out of ``analyze(Result[MetaDSpectralABF])`` (memo Sec. 4) so that
    `pysages.methods.funnel_metad_sabf` can reuse it verbatim on other per-bin force arrays of
    the same shape (``Frestr_prod``, or ``Fsum_prod + Frestr_prod``) with the same method/grid,
    rather than re-implementing the masked lstsq machinery.
    """
    model = method.model
    dims = method.grid.shape.size
    shape = (*Fsum_prod.shape[:-1], 1)
    mean_force = _onp.asarray(Fsum_prod / np.maximum(hist_prod.reshape(shape), 1))
    mean_force = mean_force.reshape(-1)
    A = vandergrad_builder(model.grid, model.exponents)(model.mesh)
    if model.is_periodic:
        A = np.hstack((np.imag(A), np.real(A)))
    A = _onp.asarray(A)
    point_mask = _onp.asarray(hist_prod).reshape(-1) >= n_min
    row_mask = _onp.repeat(point_mask, dims)
    if row_mask.sum() < A.shape[1]:  # not enough visited bins for a well-posed solve
        row_mask = _onp.ones_like(row_mask, dtype=bool)
    A_m, y_m = A[row_mask], mean_force[row_mask]
    std = float(_onp.std(y_m))
    std = 1.0 if std == 0 else std
    coeffs, *_ = _onp.linalg.lstsq(A_m, y_m / std, rcond=None)
    return Fun(np.asarray(std), np.asarray(coeffs), np.array(0.0))


def _build_fes_fn(method, fun):
    """``x -> A.max() - A`` from a `Fun` evaluated with ``method.model``; factored out of
    ``analyze(Result[MetaDSpectralABF])`` for reuse (see ``_masked_refit``)."""
    evaluate = build_evaluator(method.model)

    def fes_fn(x):
        A = evaluate(fun, x)
        return A.max() - A

    return jit(fes_fn)


def _free_energy_trapz(mesh_1d, mean_force_1d):
    """1D direct cumulative-trapezoid integral of a mean force along ``mesh_1d``; robust at a
    sharp onset (e.g. the CVDist W1 floor) where the global spectral fit can smooth over the
    sharpest bins. Factored out of ``analyze(Result[MetaDSpectralABF])`` for reuse."""
    x = _onp.asarray(mesh_1d).reshape(-1)
    mf = _onp.asarray(mean_force_1d).reshape(-1)
    order = _onp.argsort(x)
    xs, mfs = x[order], mf[order]
    avg = 0.5 * (mfs[1:] + mfs[:-1])
    f_hat_sorted = _onp.concatenate(([0.0], _onp.cumsum(avg * _onp.diff(xs))))
    fes_sorted = f_hat_sorted.max() - f_hat_sorted
    fes = _onp.empty_like(fes_sorted)
    fes[order] = fes_sorted
    return fes


@dispatch
def analyze(result: Result[MetaDSpectralABF], **kwargs):
    """
    Parameters
    ----------

    result: Result[MetaDSpectralABF]

    n_min: Optional[int]
        Bins with ``histogram < n_min`` are excluded from the analysis-time spectral refit
        (default: the method's ``fit_min_count``).

    Returns
    -------

    dict with keys:

        histogram, mean_force: JaxArray
            ``hist_prod`` and ``Fsum_prod / max(hist_prod, 1)`` (physical, metad-subtracted,
            production-only -- memo Sec. 4).

        mesh: JaxArray
            Chebyshev nodes of the sABF grid, as in `SpectralABF`.

        fun: Fun
            A spectral fit refit at analysis time on the *undamped* production means, with
            bins where ``histogram < n_min`` masked out of a small least-squares solve (*not*
            the fast precomputed-pinv `fit` used during the run, which cannot mask bins).

        free_energy, fes_fn: JaxArray, Callable
            Free energy from ``fun`` on ``mesh`` / anywhere in the domain.

        free_energy_trapz: Optional[JaxArray]
            1D only: direct cumulative-trapezoid integral of ``mean_force`` along ``mesh``,
            robust at a sharp onset (e.g. the CVDist W1 floor) where the global spectral fit
            can smooth over the sharpest bins. ``None`` for 2D.

        metad_potential: JaxArray
            ``V_G`` (well-tempered metad potential), linearly interpolated onto ``mesh``.

        heights: JaxArray
            Record of deposited hill heights (Sec. 3); a decaying tail indicates WT convergence.

        n_clipped: int
            Number of steps where the CV- or atom-space clamp fired (Sec. 8, risk 2); should be
            0 after burn-in.

    NOTE: for multiple-replica runs every key holds a list (one item per replica).
    """
    return _analyze_metad_sabf(result, **kwargs)


def _analyze_metad_sabf(result, **kwargs):
    """Body of ``analyze(Result[MetaDSpectralABF])``, factored out (unchanged behavior) so
    ``pysages.methods.funnel_metad_sabf`` can call it directly to get "everything
    `MetaDSpectralABF` returns" without going through `Result` covariant dispatch a second time
    (which, for a `Funnel_MetaDSpectralABF` result, would just call the funnel `analyze`
    override right back -- see that module's own `analyze`)."""
    method = result.method
    grid = method.grid
    metad_grid = method.metad_grid
    dims = grid.shape.size
    n_min = kwargs.get("n_min", int(method.fit_min_count))

    mesh = compute_mesh(grid)
    interp_field = build_metad_interpolator(metad_grid)

    hists, mean_forces, free_energies, funs, fes_fns = [], [], [], [], []
    trapz_fess, metad_potentials, heights_all, n_clipped_all = [], [], [], []

    transpose = grid_transposer(grid)
    d = mesh.shape[-1]

    for s in result.states:
        fun = _masked_refit(method, n_min, s.hist_prod, s.Fsum_prod)
        fes_fn = _build_fes_fn(method, fun)
        mean_force = s.Fsum_prod / np.maximum(
            np.expand_dims(s.hist_prod, s.hist_prod.ndim), 1
        )
        hists.append(transpose(s.hist_prod))
        mean_forces.append(transpose(mean_force))
        free_energies.append(transpose(fes_fn(mesh)))
        funs.append(fun)
        fes_fns.append(fes_fn)
        if dims == 1:
            trapz_fess.append(_free_energy_trapz(mesh, mean_force))
        else:
            trapz_fess.append(None)
        v_on_mesh = np.array([interp_field(s.V_G, x) for x in mesh])
        metad_potentials.append(transpose(v_on_mesh))
        heights_all.append(s.heights)
        n_clipped_all.append(int(s.n_clipped))

    ana_result = {
        "histogram": first_or_all(hists),
        "mean_force": first_or_all(mean_forces),
        "free_energy": first_or_all(free_energies),
        "mesh": transpose(mesh).reshape(-1, d).squeeze(),
        "fun": first_or_all(funs),
        "fes_fn": first_or_all(fes_fns),
        "free_energy_trapz": first_or_all(trapz_fess),
        "metad_potential": first_or_all(metad_potentials),
        "heights": first_or_all(heights_all),
        "n_clipped": first_or_all(n_clipped_all),
    }

    return numpyfy_vals(ana_result)
