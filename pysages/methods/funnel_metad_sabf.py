# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
``Funnel_MetaDSpectralABF``: `MetaDSpectralABF` (sABF + WT-metadynamics,
``pysages.methods.metad_sabf``) plus an optional ``external_force`` hook, and
``tq_upper_wall``, a ready-made ``external_force`` factory for a Calegari Andrade
(PNAS 2023, Eq. 4) -style single-ion-pair-sector wall on ``pysages.colvars.autoionization.Tq``.

Same ``external_force(data) -> (eforce, perp)`` signature convention as the reference
``Funnel_MetaDABF``/``Funnel_SpectralABF`` (``/scratch/slv5250/dpmd-ipi/PySAGES/pysages/methods/
funnel_metadabf.py``, ``/scratch/slv5250/POLYMER/07_md_engine/pysages-mlmd/pysages/methods/
funnel_sabf.py``): both apply the external force as ``bias = own_bias - eforce`` and accumulate
its CV-space projection ``restr = tsolve(Jxi, eforce)`` (no extra sign flip) into a per-bin
``Frestr``, entirely separate from the method's own ``Fsum``. Verified against those references
line by line (see the "add-back convention" section below) and against the independently
re-derived sign convention in ``/scratch/slv5250/water-committor/DECISIONS.md`` (lines 641-648,
"the real verified sign is corrected = Fsum + Frestr", toy-tested with
``scripts/toy_external_force_sign.py``).

Add-back / Frestr convention (the crux -- read this before changing anything below)
-------------------------------------------------------------------------------------
`MetaDSpectralABF`'s own estimator (module docstring there, Sec. 2.1-2.2 of the T2 design memo,
``/scratch/slv5250/SALT/plan/metad_sabf_design.md``) adds back, into ``Fsum``, exactly the
generalized force the method's OWN bias (sABF + metad) applied at the previous step. This is
what makes ``Fsum/hist`` the *physical* mean force ``-dA/dxi`` of the ensemble the system is
actually sampling: any bias that is a deterministic function of ``xi`` alone leaves the
conditional distribution ``rho(x|xi)`` unchanged once its own contribution is subtracted back
out, no matter how large or time-dependent the bias is.

``external_force`` (``eforce``) is a DIFFERENT kind of term: it is not, in general, a function of
the method's own gridded CV(s) alone -- ``tq_upper_wall`` below depends on ``Tq``, a function of
*every* oxygen's charge, not on ``CVDist``/``CVNum`` directly. Subtracting ``eforce`` back out of
``Fsum`` the same way the self-bias is subtracted would therefore be WRONG: it would try to
recover the mean force of the ensemble with NO wall at all, which is not the ensemble of
interest -- the wall is not a numerical nuisance to be cancelled, it is what DEFINES the
single-ion-pair sector whose free energy along the main CV(s) we actually want (PNAS 2023 Eq. 4:
"restrict sampling to configurations with at most one dissociated ion pair"). So:

* ``Fsum``/``Fsum_prod`` (inherited, untouched) keep adding back ONLY the self-bias, exactly as
  in plain `MetaDSpectralABF`. ``Fsum/hist`` is therefore the mean force of the RESTRAINED
  ensemble (the one the wall actually confines the system to), and the inherited
  ``free_energy``/``mean_force`` analyze() keys are the RESTRAINED FES -- the physically
  meaningful, reported quantity for a Tq-sector wall.
* ``eforce``'s own CV-space projection, ``restr = tsolve(Jxi, eforce)``, is accumulated
  SEPARATELY into new ``Frestr``/``Frestr_prod`` accumulators (the latter gated by the same
  ``burn_in`` as ``Fsum_prod``, per the design request). ``Frestr_prod/hist_prod`` is the wall's
  OWN mean projected force onto the main CV(s): ideally ~0 in well-sampled interior bins (the
  wall almost never fires there) and only nonzero near/above the wall's threshold. This is
  ``analyze()``'s ``restraint_mean_force``, and ``max_restraint_force`` (max over well-sampled
  bins) is the numeric success diagnostic Calegari Andrade-style walls are checked against: if
  it is small, the wall is doing its job of gating a rare tail rather than contaminating the
  region the FES is actually estimated over.
* ``(Fsum_prod + Frestr_prod)/hist_prod`` (``analyze()``'s ``corrected_free_energy``) recovers
  the mean force / FES as if the wall were never there -- the ``_funnelanalyze``
  (``pysages.methods.analysis``, ``/scratch/slv5250/dpmd-ipi/PySAGES``) "paper Eq. 5" convention,
  verified there against a true geometric funnel (an RMSD/path-CV restraint whose analytic
  volume correction is well defined). For a Tq-sector wall this is NOT the physically meaningful
  quantity: the "sector" itself is the point, and there is no analytic volume correction for
  "how much configuration space with 2+ ion pairs did we exclude" the way there is for a
  geometric funnel's cone/cylinder. It is provided for API parity/diagnostics only -- read
  ``free_energy``, not ``corrected_free_energy``, as the answer.

Sign check on ``eforce``: with ``tq_upper_wall``, the atom force ACTUALLY applied by the wall is
``-eforce = -(J_aux.T @ apply_restraints(...))``, i.e. identical in sign and magnitude to
`pysages.methods.aux_restraint.AuxiliaryWall`'s own ``extra`` term (same
``apply_restraints``/``CVRestraints`` primitive, same "zero below threshold, restoring harmonic
force above it" convention) -- `AuxiliaryWall` is the composable-wrapper equivalent of the same
physical wall, useful for methods without a built-in ``external_force`` hook, but it does not
track ``Frestr`` per CV bin the way this module's hook does.

``external_force=None`` (the default) makes `Funnel_MetaDSpectralABF` reduce to bit-identical
behavior with plain `MetaDSpectralABF` on every field the two share -- see
``tests/test_funnel_metad_sabf.py::test_external_force_none_is_bit_identical_to_metad_sabf``.
"""

import numpy as _onp
from jax import jit
from jax import numpy as np

from pysages.approxfun import Fun, compute_mesh
from pysages.colvars import Tq
from pysages.colvars.autoionization import DEFAULT_LAMBDA, DEFAULT_TQ_C
from pysages.colvars.core import build
from pysages.grids import build_indexer, grid_transposer
from pysages.methods.core import Result
from pysages.methods.metad_sabf import (
    MetaDSpectralABF,
    MetaDSpectralABFState,
    _analyze_metad_sabf,
    _build_fes_fn,
    _masked_refit,
)
from pysages.methods.restraints import apply_restraints
from pysages.methods.utils import numpyfy_vals
from pysages.typing import JaxArray, NamedTuple
from pysages.utils import dispatch, first_or_all, linear_solver

_N_BASE_FIELDS = len(MetaDSpectralABFState._fields)


class Funnel_MetaDSpectralABFState(NamedTuple):
    """
    `Funnel_MetaDSpectralABF` internal state.

    The first 19 fields are exactly `MetaDSpectralABFState`'s fields, in the SAME order (see
    that class's docstring for their semantics) -- ``_funnel_metad_sabf`` relies on this to
    slice ``state[:_N_BASE_FIELDS]`` back into a valid `MetaDSpectralABFState` for the inherited
    ``update``. New fields:

    Frestr, Frestr_prod: JaxArray [grid.shape, cvs.shape]
        Cumulative CV-space projection of ``eforce`` (``restr = tsolve(Jxi, eforce)``), all-time
        and burn_in-gated production respectively -- mirrors Fsum/Fsum_prod exactly, but for the
        external force instead of the method's own sABF+metad bias. See the module docstring for
        why this is accumulated SEPARATELY from Fsum rather than added back into it. Zero
        everywhere when ``external_force is None``.

    perp: JaxArray []
        Last raw auxiliary-CV value (``external_force``'s second return value), diagnostic only
        -- never gridded, binned, or estimated. Zero when ``external_force is None``.
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
    Frestr: JaxArray
    Frestr_prod: JaxArray
    perp: JaxArray

    def __repr__(self):
        return repr("PySAGES " + type(self).__name__)


class Funnel_MetaDSpectralABF(MetaDSpectralABF):
    """
    `MetaDSpectralABF` (sABF + WT-metadynamics) plus an optional ``external_force`` hook -- see
    the module docstring for the full add-back/``Frestr`` convention and its justification.

    Parameters
    ----------
    (identical to `MetaDSpectralABF`; see its docstring), plus

    external_force: Optional[Callable[[data], Tuple[JaxArray, JaxArray]]] = None
        ``external_force(data) -> (eforce, perp)``. ``eforce`` is an atom-space array
        (``[natoms, dim]`` or a flat equivalent) such that the force ACTUALLY applied to the
        atoms by the wall is ``-eforce`` (added on top of the method's own, already atom-space-
        clamped bias); ``perp`` is a diagnostic-only scalar (e.g. the raw auxiliary CV value).
        ``None`` (the default) disables the hook entirely: `Funnel_MetaDSpectralABF` then
        reduces to bit-identical behavior with plain `MetaDSpectralABF`. Passed as an ordinary
        keyword argument, e.g. ``external_force=tq_upper_wall(o_idx, h_idx, cell, 2.3, 3.0)``.

    Example
    -------
    >>> wall = tq_upper_wall(oxygen_idx, hydrogen_idx, cell, threshold=2.3, k=3.0)
    >>> method = Funnel_MetaDSpectralABF(
    ...     [CVDist(indices, cell)], grid, height=0.01, sigma=0.08, stride=250,
    ...     ngaussians=4001, deltaT=1200.0, kB=8.617333262e-5, external_force=wall,
    ... )
    """

    def build(self, snapshot, helpers, *args, **kwargs):
        self.external_force = self.kwargs.get("external_force", None)
        snapshot, base_initialize, base_update = super().build(snapshot, helpers, *args, **kwargs)
        return _funnel_metad_sabf(self, snapshot, helpers, base_initialize, base_update)


def _funnel_metad_sabf(method, snapshot, helpers, base_initialize, base_update):
    """
    Wraps the (already-built, already-jitted) `MetaDSpectralABF` ``initialize``/``update`` pair
    with the ``external_force`` hook, following the same "call the wrapped method's own update,
    then re-query and add on top" pattern as
    `pysages.methods.aux_restraint.AuxiliaryWall.build`. Unlike `AuxiliaryWall`, this also
    accumulates the external force's CV-space projection per grid bin (``Frestr``/
    ``Frestr_prod``), which requires the main CV's grid index -- hence the (cheap, redundant, but
    harmless -- ``cv``/``query`` are pure functions of ``snapshot``) second evaluation of
    ``cv(data)`` here, exactly mirroring `AuxiliaryWall`'s own re-query of ``data``.
    """
    cv = method.cv
    grid = method.grid
    dims = grid.shape.size
    burn_in = int(method.burn_in)
    external_force = method.external_force
    tsolve = linear_solver(method.use_pinv)
    get_grid_index = build_indexer(grid)
    query = helpers.query

    def initialize():
        base_state = base_initialize()
        zeros = np.zeros((*grid.shape, dims))
        return Funnel_MetaDSpectralABFState(*base_state, zeros, zeros, np.zeros(()))

    def update(snapshot, state):
        base_state = MetaDSpectralABFState(*state[:_N_BASE_FIELDS])
        new_base = base_update(snapshot, base_state)

        data = query(snapshot)
        xi, Jxi = cv(data)
        I_xi = get_grid_index(xi)
        p = data.momenta

        if external_force is None:
            eforce = np.zeros_like(new_base.bias)
            perp = np.zeros(())
        else:
            eforce_raw, perp_raw = external_force(data)
            eforce = np.reshape(eforce_raw, new_base.bias.shape)
            perp = np.reshape(perp_raw, ())

        # Actually-applied atom force = own (already atom-clamped) bias minus the external
        # force; `force_applied` (inside `new_base`, used for NEXT step's Fsum add-back) is left
        # untouched -- it was already computed by `base_update` from the self-bias ALONE. This
        # is the crux of the module docstring's convention: Fsum only ever "sees" the method's
        # own bias, never `eforce`.
        bias = new_base.bias - eforce

        # CV-space projection of the external force, accumulated separately (never into Fsum).
        restr = tsolve(Jxi, eforce.reshape(p.shape))
        in_prod = new_base.ncalls > burn_in
        Frestr = state.Frestr.at[I_xi].add(restr)
        Frestr_prod = state.Frestr_prod.at[I_xi].add(np.where(in_prod, restr, 0.0))

        new_base = new_base._replace(bias=bias)
        return Funnel_MetaDSpectralABFState(*new_base, Frestr, Frestr_prod, perp)

    return snapshot, initialize, jit(update)


@dispatch
def analyze(result: Result[Funnel_MetaDSpectralABF], **kwargs):
    """
    Parameters
    ----------

    result: Result[Funnel_MetaDSpectralABF]

    n_min: Optional[int]
        As in `analyze(Result[MetaDSpectralABF])`: bins with ``histogram < n_min`` are excluded
        from the analysis-time spectral refits of ``free_energy``, ``restraint_energy`` and
        ``corrected_free_energy`` alike (default: the method's ``fit_min_count``).

    restraint_n_min: Optional[int]
        Bins with ``hist_prod <= restraint_n_min`` are excluded from the ``max_restraint_force``
        diagnostic (default: the method's ``N``, per the design request "bins with hist > N").

    Returns
    -------

    dict: everything `analyze(Result[MetaDSpectralABF])` returns (``histogram``, ``mean_force``,
    ``mesh``, ``fun``, ``free_energy``, ``fes_fn``, ``free_energy_trapz``, ``metad_potential``,
    ``heights``, ``n_clipped`` -- see that docstring; ``free_energy``/``mean_force`` there are
    the RESTRAINED FES/mean force, see the module docstring), plus:

        restraint_mean_force: JaxArray
            ``Frestr_prod / max(hist_prod, 1)`` -- the wall's own mean projected force onto the
            main CV(s). Should be ~0 in well-sampled interior bins for a well-placed wall.

        restraint_energy: JaxArray
            Spectral-fit free-energy integral of ``restraint_mean_force`` alone.

        corrected_free_energy: JaxArray
            Spectral-fit free-energy integral of
            ``(Fsum_prod + Frestr_prod) / max(hist_prod, 1)`` -- the FES as if the wall were
            never there. Only physically meaningful for a true geometric funnel with an
            analytic correction; for a Tq-sector wall (`tq_upper_wall`) this is a diagnostic
            only -- ``free_energy`` (the RESTRAINED FES) is the physically meaningful quantity
            to report. See the module and `tq_upper_wall` docstrings.

        max_restraint_force: float (list, multi-replica)
            ``max(|Frestr_prod/hist_prod|)`` over bins with ``hist_prod > restraint_n_min``
            (``NaN`` if no bin qualifies). Small (~0) is the success diagnostic for a
            well-placed wall.

    NOTE: this analyze() deliberately does NOT emit a key literally named ``force_corrected``
    (note the word order -- distinct from ``corrected_free_energy`` above): autofes's
    ``combine.py.j2`` template prefers a ``force_corrected`` key over ``mean_force`` when
    combining windows into a global FES, and for this method the RESTRAINED ``mean_force`` is
    the one that should be combined across windows (see the module docstring), not the
    wall-corrected estimate.

    NOTE: for multiple-replica runs every key holds a list (one item per replica), exactly like
    the base `MetaDSpectralABF` analyze().
    """
    method = result.method
    n_min = kwargs.get("n_min", int(method.fit_min_count))
    n_restraint_min = kwargs.get("restraint_n_min", int(method.N))

    base = _analyze_metad_sabf(result, **kwargs)

    grid = method.grid
    mesh = compute_mesh(grid)
    transpose = grid_transposer(grid)

    restraint_mean_forces, restraint_energies = [], []
    corrected_free_energies, max_restraint_forces = [], []

    for s in result.states:
        hist_prod = s.hist_prod
        Frestr_prod = s.Frestr_prod
        Fsum_prod = s.Fsum_prod

        shape = (*Frestr_prod.shape[:-1], 1)
        restr_force = Frestr_prod / np.maximum(hist_prod.reshape(shape), 1)
        restraint_mean_forces.append(transpose(restr_force))

        fun_r = _masked_refit(method, n_min, hist_prod, Frestr_prod)
        fes_r = _build_fes_fn(method, fun_r)
        restraint_energies.append(transpose(fes_r(mesh)))

        fun_c = _masked_refit(method, n_min, hist_prod, Fsum_prod + Frestr_prod)
        fes_c = _build_fes_fn(method, fun_c)
        corrected_free_energies.append(transpose(fes_c(mesh)))

        hist_np = _onp.asarray(hist_prod)
        rf_np = _onp.asarray(restr_force)
        well_sampled = hist_np > n_restraint_min
        if well_sampled.any():
            max_restraint_forces.append(float(_onp.max(_onp.abs(rf_np[well_sampled]))))
        else:
            max_restraint_forces.append(float("nan"))

    base.update(
        {
            "restraint_mean_force": first_or_all(restraint_mean_forces),
            "restraint_energy": first_or_all(restraint_energies),
            "corrected_free_energy": first_or_all(corrected_free_energies),
            "max_restraint_force": first_or_all(max_restraint_forces),
        }
    )
    return numpyfy_vals(base)


# -------------------------------------------------------------------------------------------
# tq_upper_wall: a ready-made external_force factory for a Calegari Andrade PNAS 2023 Eq. 4
# -style single-ion-pair-sector wall.
# -------------------------------------------------------------------------------------------


def tq_upper_wall(o_idx, h_idx, cell, threshold, k, lam=DEFAULT_LAMBDA, c=DEFAULT_TQ_C):
    """
    External-force factory: an upper-half harmonic wall on `pysages.colvars.autoionization.Tq`
    (Calegari Andrade, Prendergast et al., PNAS 2023, Eq. 4), ready to pass as
    ``Funnel_MetaDSpectralABF(..., external_force=tq_upper_wall(...))``.

    Zero force for ``Tq <= threshold``; above it, a harmonic restoring force
    ``k * (Tq - threshold)`` pushing Tq back down -- the same one-sided-wall convention as
    `pysages.methods.restraints.apply_restraints`/`CVRestraints`, i.e. physically identical to
    ``pysages.methods.aux_restraint.AuxiliaryWall(base, Tq(...), threshold, k, side="upper")``.
    Use `AuxiliaryWall` instead if the method you want to wall does not have a built-in
    ``external_force`` hook; use THIS factory with `Funnel_MetaDSpectralABF`, whose hook
    additionally tracks the wall's own projected mean force (``Frestr``) per main-CV bin, which
    `AuxiliaryWall` does not.

    Physics (see the module and `Funnel_MetaDSpectralABF` docstrings for the full derivation):
    the wall defines the single-ion-pair-sector ensemble the main CV's free energy is computed
    in (PNAS 2023 Eq. 4: "at most one simultaneous H3O+/OH- pair"), so the RESTRAINED free
    energy (`Funnel_MetaDSpectralABF`'s inherited ``free_energy``/``mean_force``) is the
    physically meaningful, reported quantity -- NOT ``corrected_free_energy`` (meaningful only
    for a true geometric funnel with an analytic correction for the confining potential, which a
    Tq-sector wall is not: removing it changes which physical ensemble is being described, not
    just a bias to subtract away). ``Frestr`` close to zero in well-sampled interior bins
    (``analyze()``'s ``restraint_mean_force``/``max_restraint_force``) is the success
    diagnostic: it means the wall only clips the rare high-Tq tail and barely perturbs sampling
    anywhere the main CV's FES is actually being estimated.

    Parameters
    ----------
    o_idx, h_idx: list[int]
        Water oxygen / hydrogen atom indices, as for `Tq`/`CVDist`/`CVNum` (salt ions excluded;
        see the ``pysages.colvars.autoionization`` module docstring, "Salt ions").
    cell: array-like, shape (3, 3) or (3,)
        Fixed simulation cell (NVT), same convention as `Tq`.
    threshold: float
        Wall location on Tq. See `tq_threshold_from_baseline` (an analytic choice) and
        `calibrate_tq_threshold` (an empirical one, from an unbiased trajectory) below.
    k: float
        Upper-wall spring constant, in force units per unit Tq. Keep it in the same modest range
        (about 3-4) as the main CV's own grid-wall ``restraints``: a much stiffer wall risks the
        same blow-ups documented for over-stiff CVRestraints springs
        (``/scratch/slv5250/water-committor/DECISIONS.md``, "BLOWUP root cause = wall spring
        constant units").
    lam, c: float, default `DEFAULT_LAMBDA`, `DEFAULT_TQ_C`
        Passed through to `Tq` (softmax sharpness, sqrt regularizer); keep these equal to
        whatever the run's main CV (e.g. `CVDist`) uses, if any, so ``Tq``'s per-oxygen charges
        ``q_o`` are computed the same way.

    Returns
    -------
    Callable[[data], Tuple[JaxArray, JaxArray]]
        ``external_force(data) -> (eforce, perp)`` (see `Funnel_MetaDSpectralABF`'s docstring
        for the exact ``eforce`` sign convention); ``perp`` is the current Tq value.
    """
    tq_cv = Tq([list(o_idx), list(h_idx)], cell, lam=lam, c=c)
    aux_apply = build(tq_cv, differentiate=True)
    lo, hi, kl, ku = -np.inf, float(threshold), 0.0, float(k)

    def external_force(data):
        xi_aux, J_aux = aux_apply(data)
        f_cv = apply_restraints(lo, hi, kl, ku, xi_aux.reshape(1))
        eforce = (J_aux.T @ f_cv).reshape(-1)
        perp = xi_aux.reshape(())
        return eforce, perp

    return external_force


def tq_baseline(n_oxygen, c=DEFAULT_TQ_C):
    """
    ``N_O * sqrt(c)``: the fully-associated-water baseline of `Tq` (every per-oxygen charge
    ``q_o ~= 0``; see ``pysages.colvars.autoionization.tq_value``'s docstring).
    """
    return float(n_oxygen) * float(_onp.sqrt(c))


def tq_threshold_from_baseline(n_oxygen, delta, c=DEFAULT_TQ_C):
    """
    ``tq_baseline(n_oxygen, c) + delta``: an analytic choice of `tq_upper_wall`'s ``threshold``,
    ``delta`` units of Tq above the fully-associated baseline. A well-separated single ion pair
    gives ``Tq ~= 2 + baseline`` (one oxygen at ``q ~= +1``, one at ``q ~= -1``); ``delta ~= 2``
    sits just above that, wide enough to let one pair form freely while still gating a second,
    simultaneous pair (``Tq ~= 4 + baseline``).
    """
    return tq_baseline(n_oxygen, c) + float(delta)


def calibrate_tq_threshold(tq_trajectory, percentile=99.9, margin=0.05):
    """
    Empirically calibrate `tq_upper_wall`'s ``threshold`` from an UNBIASED (or main-CV-only
    biased) trajectory's Tq time series, so the wall sits just above Tq's natural fluctuation
    ceiling in the "at most one ion pair" regime rather than clipping physical fluctuations.

    Parameters
    ----------
    tq_trajectory: array-like
        Tq values sampled along the reference run.
    percentile: float, default 99.9
        Percentile of the empirical Tq distribution used as the base level.
    margin: float, default 0.05
        Extra Tq units added above that percentile, so the wall practically never fires in the
        bulk of sampling -- only on the rare tail toward a second, simultaneous ion pair.

    Returns
    -------
    float: recommended ``threshold`` for `tq_upper_wall`.
    """
    values = _onp.asarray(tq_trajectory, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("tq_trajectory is empty")
    return float(_onp.percentile(values, percentile) + margin)
