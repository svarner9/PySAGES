# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
``AuxiliaryWall``: a one-sided (or two-sided) harmonic wall on a collective variable that is
*not* one of a `SamplingMethod`'s sampled/biased dimensions.

Motivation
----------
`GriddedSamplingMethod.restraints` (`pysages.methods.restraints.CVRestraints`) already gives
every ABF-family method a per-CV wall, but it is indexed by *grid* dimension: it only ever fires
through the out-of-bounds branch of that method's own grid (see e.g.
`pysages.methods.metad_sabf._metad_sabf`'s ``out_of_bounds`` branch), one entry per sampled CV.
An auxiliary safety CV that a method never grids, bins, or estimates a mean force for -- e.g.
``Tq`` (`pysages.colvars.autoionization.Tq`) used to cap the number of simultaneous ion pairs in
a CVDist/CVNum global-CV run -- does not fit that shape: extending `CVRestraints` to carry it
would mean threading an extra CV dimension through the grid indexer, histogram, and estimator of
*every* ABF-family method for a quantity that method should never otherwise see.

Instead, `AuxiliaryWall` wraps an already-constructed method and adds a plain harmonic wall
force in atom space, after the wrapped method's own `update`. This is the smaller, more
general change: every `SamplingMethod.build()` is documented (`pysages/methods/__init__.py`) to
return ``(snapshot, initialize, update)`` with ``state.bias`` as a required field, so a wrapper
that only ever *reads* ``data`` (already queried once per step) and *adds* to that field
composes with ABF, SpectralABF, MetaDSpectralABF, Metadynamics, a harmonic bias, or any future
method -- without needing to know its internal state layout, and without adding a grid
dimension, a histogram bin, or an estimator branch that method's ``analyze()`` would have to
special-case away.

The wall itself reuses `pysages.methods.restraints.apply_restraints` and `CVRestraints`
verbatim (the same primitive every ABF-family method already uses for its own grid walls), so
its force convention (magnitude, sign, "zero below/above threshold") is identical to theirs.
"""

from jax import numpy as np

from pysages.colvars.core import build
from pysages.methods.core import SamplingMethod
from pysages.methods.restraints import CVRestraints, apply_restraints, canonicalize


class AuxiliaryWall(SamplingMethod):
    """
    Wraps ``method`` and adds a one-sided (or two-sided) harmonic wall on ``aux_cv``, a
    collective variable that ``method`` itself never sees.

    Parameters
    ----------

    method: SamplingMethod
        Any already-constructed sampling method (`ABF`, `SpectralABF`, `MetaDSpectralABF`,
        `Metadynamics`, `HarmonicBias`, ...). Its own bias is left untouched; this wrapper only
        adds to it.

    aux_cv: CollectiveVariable
        The auxiliary CV to wall off, e.g. ``Tq(indices, cell)``. Not gridded, binned, or
        estimated by ``method`` -- only ever contributes a harmonic force.

    threshold: float
        Wall location.

    k: float
        Spring constant (force units per unit of ``aux_cv``, squared-then-differentiated same
        as `CVRestraints`). ``0`` disables that side (see `CVRestraints.canonicalize`).

    side: str = "upper"
        ``"upper"``: force is 0 for ``aux_cv <= threshold``, harmonic and restoring above it
        (e.g. a cap on `Tq` to forbid more than one simultaneous ion pair). ``"lower"``: the
        mirror image. ``"both"``: use ``threshold=(lower, upper)`` and ``k=(kl, ku)`` instead
        (a two-sided box on the auxiliary CV).

    Example
    -------
    >>> tq = Tq([oxygen_idx, hydrogen_idx], cell)
    >>> base = MetaDSpectralABF(cvs, grid, height=0.01, sigma=0.08, stride=250,
    ...                         ngaussians=4001, deltaT=1200.0, kB=8.617333262e-5)
    >>> method = AuxiliaryWall(base, tq, threshold=2.2, k=100 / 96.485, side="upper")
    """

    def __init__(self, method: SamplingMethod, aux_cv, threshold, k, side="upper"):
        if side not in ("upper", "lower", "both"):
            raise ValueError('`side` must be one of "upper", "lower", "both"')
        self.method = method
        self.aux_cv = aux_cv
        self.threshold = threshold
        self.k = k
        self.side = side
        self.snapshot_flags = set(method.snapshot_flags)
        # Not read by the mlmd backend, but kept consistent for other backends (hoomd, lammps)
        # that check `requires_box_unwrapping`, and for anything that inspects `.cvs`.
        self.cvs = list(getattr(method, "cvs", [])) + [aux_cv]
        self.requires_box_unwrapping = (
            getattr(method, "requires_box_unwrapping", False) or aux_cv.requires_box_unwrapping
        )

        if side == "both":
            lower, upper = threshold
            kl, ku = k
        elif side == "upper":
            lower, upper, kl, ku = -np.inf, threshold, 0.0, k
        else:
            lower, upper, kl, ku = threshold, np.inf, k, 0.0
        self._wall = canonicalize(CVRestraints(lower=[lower], upper=[upper], kl=[kl], ku=[ku]), [aux_cv])
        self._aux_cv_fn = build(aux_cv, differentiate=True)

    def build(self, snapshot, helpers, *args, **kwargs):
        # `self.method.build(...)` already returns `update` wrapped by
        # `pysages.methods.core.generalize`, i.e. with signature `update(snapshot, state)`
        # (it calls `helpers.query(snapshot)` internally to get `data` -- see `generalize`).
        # This wrapper must match that external signature, and re-query `data` itself to
        # evaluate the auxiliary CV.
        snapshot, initialize, update = self.method.build(snapshot, helpers, *args, **kwargs)
        aux_cv = self._aux_cv_fn
        lo, hi, kl, ku = self._wall
        query = helpers.query

        def wrapped_update(snapshot, state):
            state = update(snapshot, state)
            data = query(snapshot)
            xi_aux, J_aux = aux_cv(data)
            force = apply_restraints(lo, hi, kl, ku, xi_aux.reshape(1))
            extra = np.reshape(-J_aux.T @ force, state.bias.shape)
            return state._replace(bias=state.bias + extra)

        return snapshot, initialize, wrapped_update

    def __getstate__(self):
        return {
            "method": self.method,
            "aux_cv": self.aux_cv,
            "threshold": self.threshold,
            "k": self.k,
            "side": self.side,
        }

    def __setstate__(self, state):
        self.__init__(**state)
