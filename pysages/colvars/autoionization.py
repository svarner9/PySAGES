# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Autoionization collective variables for water: ``CVNum``, ``CVDist``, and ``Tq``.

Both drive water self-dissociation / recombination (H2O + H2O <-> H3O+ + OH-)
in a box of water (optionally with dissolved salt ions, which are excluded
from these CVs -- see "Salt ions" below).  They are a straight port of the
CVs used by the SALT project's pre-mlmd workflow, whose canonical source is

    - CVnum:  ``compute_ni``/``compute_cv_num`` in
      ``cvNumCollectiveVariableOld.py``, class ``CvNumCollectiveVariableOld``
    - CVdist: ``_compute_cv_dist`` in ``cvDistCollectiveVariableGem.py``,
      class ``CvDistCollectiveVariableGem``

both at ``/scratch/slv5250/dpmd-ipi/PySAGES/pysages/colvars/`` -- the exact
path named in the header comment of ``SALT/config_gen/run_biased_md.py``,
which is the SALT driver that imports these two classes by name. Those two
files are *untracked* working-tree additions in that checkout (``git status``
there shows them as ``??``, not part of any commit), which is also why the
pip-installed ``pysages`` wheels in the ``base``/``analysis`` conda
environments (built from a git-tracked snapshot) do NOT contain them --
``run_biased_md.py`` only sees these classes because it runs with that
checkout's source directory itself on ``sys.path``/``PYTHONPATH`` ahead of
the installed wheel. Lightly-hardened copies of the same formulas (adding a
clip and a safe-division guard to CVnum) also live at
``/scratch/slv5250/DFT/autofes/examples/{ase_2d,lammps_2d}/cvs/{cvdist,cvnum}.py``.
All were used for value/gradient parity testing (``tests/test_autoion_cvs.py``).

Definitions
-----------
For a set of ``N_O`` water oxygens and ``N_H`` water hydrogens (salt ions
excluded -- see below), all distances taken under the minimum image
convention (MIC):

``CVNum`` (coordination-defect sum)::

    n_i    = sum_H s(r_OH)                          per-oxygen H count
    s(r)   = (1 - x^16) / (1 - x^56),  x = min(r / R0, clip)
    CVNum  = sum_O (n_i - 2)^2

  R0 = 1.32 Å is the O-H switching-function inflection radius; for an intact
  water molecule each O binds exactly 2 H "tightly" (s ~= 1 for the covalent
  O-H, s ~= 0 for everything else) so n_i ~= 2 and CVNum ~= 0.  An H3O+ (n_i
  ~= 3) or OH- (n_i ~= 1) unbalances the sum.

``CVDist`` (environment-weighted O-O distance)::

    q_i     = sum_H [ exp(-lam r_OH) / sum_{O'} exp(-lam r_O'H) ] - 2
    CVDist  = - sum_{i<k} q_i q_k r_OO(i, k)

  lam = 8 / Å.  The bracket is a softmax *of each H over all oxygens*: for
  every hydrogen it distributes weight 1 among the oxygens, concentrated on
  whichever is closest.  Summed over H for an intact water oxygen (2 "own"
  H each contributing ~1) this gives q_i ~= 0; an OH- (only ~1 H "assigned")
  gives q_i ~= -1 and an H3O+ (~3 H) gives q_i ~= +1.  CVDist is then (minus)
  the charge-weighted sum of O-O distances, which is ~= +r_OO for a well
  separated H3O+/OH- pair (q_i q_k ~= (+1)(-1) = -1) and ~= 0 for intact
  water.

Numerical hardening (float32 safety)
-------------------------------------
``x**56`` overflows ``float32`` (max ~3.4e38) once ``x`` = r/R0 exceeds about
4.87 -- which happens for essentially every non-bonded O-H pair in any
water box bigger than a couple of R0's, i.e. always.  The raw
``CvNumCollectiveVariableOld`` has no such clip and is only usable in
float64 (as ``run_biased_md.py`` itself notes: it forces ``float64`` for the
whole CV/bias machinery specifically because of this and the CVdist
cancelling-sum issue below).  Clipping ``x`` at ``clip=3.5`` (this module's
default, matching the ``autofes`` "hardened" reference) keeps ``x**56`` <=
3.5**56 ~= 3e30, comfortably inside float32 range, **without changing the CV
value**: physically, an O-H pair at r/R0 > 3.5 (r > 4.6 Å) should already
contribute ~0 to a coordination number, and ``s(clip)`` at the clip point
already IS ~1e-22 -- clipping a term that small can only move the CV value by
an amount far below any float32 (let alone float64) precision floor. This is
verified empirically in ``tests/test_autoion_cvs.py`` by cross-checking the
clipped port against the raw (unclipped) reference class in float64 on real
frames.

``s(R0)`` (``x == 1``) is a removable 0/0 singularity; its true limit
(L'Hopital) is 2/7, but both this port and the ``autofes`` hardened
reference instead map the neighbourhood of the singularity to 0 (a ``where``
guard on ``|1 - x**56| < 1e-12``) purely so the *gradient* stays finite
there.  This is a deliberate parity choice with the reference rather than a
"fix": hitting ``x == 1`` to bit-for-bit floating point equality from real MD
coordinates is a measure-zero event that is not observed in any of the test
frames.

``CVDist`` has no analogous overflow risk (it only ever exponentiates
non-positive arguments), but it *is*, as ``run_biased_md.py`` puts it, "a
large cancelling sum over O-O pairs built from ratios of exponentials --
worst case for float32"; ``tests/test_autoion_cvs.py`` quantifies the
float32-vs-float64 gap directly rather than papering over it, and production
biasing of CVDist should keep the collective-variable machinery in float64
(``jax.config.update("jax_enable_x64", True)``), exactly as the SALT driver
already does.

PBC / cell convention
----------------------
Distances use a general (not necessarily orthorhombic) minimum image: the
cell can be any 3x3 matrix, fractional coordinates are rounded to the
nearest image and mapped back to Cartesian, exactly as the reference
``dist_mic``/``dist_mic_matrix`` helpers do.  This is exact for orthorhombic
cells and a good approximation for mildly triclinic ones (the usual caveat
of round-based MIC for very skewed cells applies equally to the reference).

The cell is supplied once, at construction time, and closed over by
``function`` -- the same convention the reference classes use (``self.cell =
cell``) and the same one this fork's ``NematicOrderParameter`` CV uses (see
``pysages/colvars/nematic.py``) on the ``mlmd-backend`` branch as committed
(``98d5288``).  A separate, not-yet-merged line of work on that branch's
live checkout adds a ``requires_cell`` flag so a CV's ``function`` can
instead receive the *current* per-step cell from the backend (relevant for
an NPT barostat); that mechanism is not present in the commit this module is
built against, so it is not used here.  It is also not needed for the SALT
production plan: CVDist is not invariant under uniform box scaling (R0 and
1/lam are fixed lengths, and O-O distances scale while the softmax does not
scale the same way), so PySAGES biasing along either CV must run NVT, where
the cell never changes.  Should the platform later support scale-invariant
NPT with a dynamically-supplied cell, subclassing to add ``requires_cell``
is a small extension of this module.

Salt ions
---------
Both CVs only ever see two atom groups, water oxygens and water hydrogens;
there is no notion of "everything else" in the formulas. To use these CVs on
a system that also contains dissolved salt (Li+, Ca2+, Cl-, I-, ...), simply
build the ``[oxygen_indices, hydrogen_indices]`` index groups from that
system's water atoms only (e.g. by element symbol) -- the convenience
constructor :meth:`_WaterAutoionizationCV.from_atoms` does exactly this,
selecting every ``O`` and ``H`` atom and silently excluding anything else.
"""

from jax import numpy as np

from pysages.colvars.core import TwoPointCV

# R0: O-H switching-function inflection radius (Å).
DEFAULT_R0 = 1.32
# x = r / R0 is clipped at this value before raising to the 56th power, the
# float32-overflow guard described in the module docstring.
DEFAULT_CLIP = 3.5
# (numerator power, denominator power) of the rational switching function.
DEFAULT_POWERS = (16, 56)
# lambda: per-H softmax sharpness over oxygens (1/Å).
DEFAULT_LAMBDA = 8.0
# c: regularizer under the sqrt in Tq = sum_o sqrt(q_o**2 + c) (Calegari Andrade et al.,
# PNAS 2023, Eq. 4), keeping the |q| kink at q=0 smooth/differentiable.
DEFAULT_TQ_C = 0.01


def minimum_image_vectors(p1, p2, cell):
    """
    Pairwise minimum-image displacement vectors ``p1[i] - p2[j]``.

    Parameters
    ----------
    p1: jax.Array, shape (n1, 3)
    p2: jax.Array, shape (n2, 3)
    cell: jax.Array, shape (3, 3)
        Cell (row) vectors; need not be orthorhombic.

    Returns
    -------
    jax.Array, shape (n1, n2, 3)
    """
    cell = np.asarray(cell)
    cell_inv = np.linalg.inv(cell)
    d = p1[:, None, :] - p2[None, :, :]
    d_frac = d @ cell_inv
    d_frac = d_frac - np.round(d_frac)
    return d_frac @ cell


def minimum_image_distances(p1, p2, cell, eps=1e-30):
    """
    Pairwise minimum-image distances, shape ``(len(p1), len(p2))``.

    ``eps`` is added under the square root purely to keep ``d(sqrt)/dr``
    finite in case ``p1`` and ``p2`` ever coincide; it is negligible
    (``sqrt(eps) = 1e-15``) next to any physically real O-H distance.
    """
    d = minimum_image_vectors(p1, p2, cell)
    return np.sqrt(np.sum(d**2, axis=-1) + eps)


def _self_distances(p, cell):
    """
    Symmetric pairwise minimum-image distances of ``p`` against itself.

    Unlike :func:`minimum_image_distances`, the diagonal (self-distance) is
    forced to an *exact* zero via a ``where`` guard rather than perturbed by
    ``eps``: it feeds ``CVDist``, where the diagonal term must vanish exactly
    (an oxygen does not interact with itself), and ``where`` also keeps its
    gradient finite at r = 0 (the same pattern the reference
    ``cvDistCollectiveVariableGem.py`` uses for its O-O block).
    """
    d = minimum_image_vectors(p, p, cell)
    sq = np.sum(d**2, axis=-1)
    tiny = sq < 1e-20
    safe_sq = np.where(tiny, 1.0, sq)
    return np.where(tiny, 0.0, np.sqrt(safe_sq))


def switching_function(r, R0=DEFAULT_R0, clip=DEFAULT_CLIP, powers=DEFAULT_POWERS):
    """
    Rational switching function ``s(r) = (1 - x**p) / (1 - x**q)`` with
    ``x = min(r / R0, clip)``, ``(p, q) = powers``.

    See the module docstring ("Numerical hardening") for why ``x`` is
    clipped and how the removable singularity at ``x == 1`` is handled.
    """
    p, q = powers
    x = np.minimum(r / R0, clip)
    num = 1 - x**p
    den = 1 - x**q
    singular = np.abs(den) < 1e-12
    safe_den = np.where(singular, 1.0, den)
    return np.where(singular, 0.0, num / safe_den)


def cv_num_value(
    p_oxygen, p_hydrogen, cell, R0=DEFAULT_R0, clip=DEFAULT_CLIP, powers=DEFAULT_POWERS
):
    """CVNum = sum_O (n_i - 2)**2; see the module docstring for ``n_i``."""
    r_oh = minimum_image_distances(p_oxygen, p_hydrogen, cell)  # (n_O, n_H)
    s = switching_function(r_oh, R0, clip, powers)
    n_i = np.sum(s, axis=1)  # per-oxygen H count, summed over hydrogens
    return np.sum((n_i - 2.0) ** 2)


def cv_dist_value(p_oxygen, p_hydrogen, cell, lam=DEFAULT_LAMBDA):
    """CVDist = -sum_{i<k} q_i q_k r_OO(i, k); see the module docstring."""
    qi = _charge_deficits(p_oxygen, p_hydrogen, cell, lam)  # (n_O,)

    r_oo = _self_distances(p_oxygen, cell)  # (n_O, n_O), exact zero diagonal
    qi_outer = qi[:, None] * qi[None, :]
    # The i<k sum equals half the full symmetric matrix (the diagonal is
    # exactly zero, and off-diagonal terms are pairwise equal), matching
    # `cvDistCollectiveVariableGem.py`'s `sum(...) / 2` exactly.
    return -0.5 * np.sum(qi_outer * r_oo)


def _charge_deficits(p_oxygen, p_hydrogen, cell, lam=DEFAULT_LAMBDA):
    """Per-oxygen smooth Grifoni charge ``q_i`` (shared by ``CVDist`` and ``Tq``, see the
    ``Tq`` docstring below); ``cell`` only enters through the O-H minimum-image distances."""
    r_oh = minimum_image_distances(p_oxygen, p_hydrogen, cell)  # (n_O, n_H)
    exp_oh = np.exp(-lam * r_oh)
    sum_over_oxygens = np.sum(exp_oh, axis=0)  # (n_H,), per-H normalization
    return np.sum(exp_oh / sum_over_oxygens[None, :], axis=1) - 2.0  # (n_O,)


def tq_value(p_oxygen, p_hydrogen, cell, lam=DEFAULT_LAMBDA, c=DEFAULT_TQ_C):
    """
    ``Tq = sum_o sqrt(q_o**2 + c)``, a smooth surrogate for ``sum_o |q_o|`` built from the same
    per-oxygen Grifoni charges ``q_o`` as ``CVDist`` (Calegari Andrade, Prendergast et al.,
    PNAS 2023, Eq. 4). ``Tq ~= 2`` when a single H3O+/OH- pair is well separated (one oxygen at
    ``q ~= +1``, one at ``q ~= -1``, each contributing ``sqrt(1 + c) ~= 1``) plus a small
    ``N_O * sqrt(c)`` baseline from the remaining intact-water oxygens (``q ~= 0``); it stays
    small (``~= N_O * sqrt(c)``) when the box is fully associated. ``c`` regularizes the
    ``|x|`` kink at ``q = 0`` so the gradient stays finite and JIT-differentiable everywhere.
    """
    qi = _charge_deficits(p_oxygen, p_hydrogen, cell, lam)
    return np.sum(np.sqrt(qi**2 + c))


def _water_groups(atoms):
    """[oxygen_indices, hydrogen_indices], cell) from an ``ase.Atoms`` frame,
    silently excluding any other element (salt ions)."""
    symbols = list(atoms.get_chemical_symbols())
    oxygen = [i for i, s in enumerate(symbols) if s == "O"]
    hydrogen = [i for i, s in enumerate(symbols) if s == "H"]
    if not oxygen or not hydrogen:
        raise ValueError("frame has no O/H atoms to build a water autoionization CV from")
    cell = np.asarray(atoms.get_cell()[:], dtype=float)
    return oxygen, hydrogen, cell


class _WaterAutoionizationCV(TwoPointCV):
    """
    Shared plumbing for :class:`CVNum` and :class:`CVDist`.

    Parameters
    ----------
    indices: [oxygen_indices, hydrogen_indices]
        Two groups: every water oxygen index, every water hydrogen index.
        Salt ions (Li, Ca, Cl, I, ...) must simply be left out of both
        groups -- see the module docstring ("Salt ions"). Prefer
        :meth:`from_atoms` to build these automatically from an
        ``ase.Atoms`` frame.
    cell: array-like, shape (3, 3) or (3,)
        The (fixed) simulation cell. A shape-``(3,)`` input is treated as
        the diagonal of an orthorhombic cell. Fixed at construction time:
        these CVs are for NVT (see the module docstring, "PBC / cell
        convention").
    """

    def __init__(self, indices, cell):
        super().__init__(indices)
        cell = np.asarray(cell, dtype=float)
        self.cell = np.diag(cell) if cell.ndim == 1 else cell
        # Neither CV is invariant under uniform box scaling (R0 and 1/lam are
        # fixed lengths), so an isotropic-NPT bias along either is unsound.
        # This flag is inert on the `mlmd-backend` commit this module targets
        # (no consumer reads it yet) but documents the CV's actual physics
        # and costs nothing; see the module docstring.
        self.is_isotropic_scale_invariant = False

    @classmethod
    def from_atoms(cls, atoms, **kwargs):
        """Build directly from an ``ase.Atoms`` frame (uses its current cell
        and every O/H atom; other elements -- salt ions -- are excluded)."""
        oxygen, hydrogen, cell = _water_groups(atoms)
        return cls([oxygen, hydrogen], cell, **kwargs)


class CVNum(_WaterAutoionizationCV):
    """
    Coordination-defect sum ``CVNum = sum_O (n_i - 2)**2``.

    ``CVNum ~= 0`` for intact water; a single autoionized H3O+/OH- pair
    raises it to ``~= (3-2)**2 + (1-2)**2 = 2``.  See the module docstring
    for the full definition, the float32 clipping, and provenance.

    Parameters
    ----------
    indices: [oxygen_indices, hydrogen_indices]
    cell: array-like, shape (3, 3) or (3,)
    R0: float, default 1.32 (Å)
        O-H switching-function inflection radius.
    clip: float, default 3.5
        Clip on ``r / R0`` before the 56th power (float32-overflow guard;
        physically inert -- see the module docstring).
    powers: (int, int), default (16, 56)
        ``(numerator, denominator)`` exponents of the rational switch.
    """

    def __init__(self, indices, cell, R0=DEFAULT_R0, clip=DEFAULT_CLIP, powers=DEFAULT_POWERS):
        super().__init__(indices, cell)
        self.R0 = float(R0)
        self.clip = float(clip)
        self.powers = (int(powers[0]), int(powers[1]))

    @property
    def function(self):
        cell, R0, clip, powers = self.cell, self.R0, self.clip, self.powers
        return lambda p1, p2: cv_num_value(p1, p2, cell, R0, clip, powers)


class CVDist(_WaterAutoionizationCV):
    """
    Environment-weighted O-O distance ``CVDist = -sum_{i<k} q_i q_k r_OO``.

    ``CVDist ~= 0`` for intact water; a well separated H3O+/OH- pair at O-O
    distance ``d`` gives ``CVDist ~= d`` (the two ions carry ``q ~= +1`` and
    ``q ~= -1``). See the module docstring for the full definition and
    provenance.

    Parameters
    ----------
    indices: [oxygen_indices, hydrogen_indices]
    cell: array-like, shape (3, 3) or (3,)
    lam: float, default 8.0 (1/Å)
        Softmax sharpness of each hydrogen's assignment over oxygens.
    """

    def __init__(self, indices, cell, lam=DEFAULT_LAMBDA):
        super().__init__(indices, cell)
        self.lam = float(lam)

    @property
    def function(self):
        cell, lam = self.cell, self.lam
        return lambda p1, p2: cv_dist_value(p1, p2, cell, lam)


class Tq(_WaterAutoionizationCV):
    """
    Total autoionization order parameter ``Tq = sum_o sqrt(q_o**2 + c)``, built from the same
    per-oxygen smooth Grifoni charges ``q_o`` as ``CVDist`` (Calegari Andrade, Prendergast
    et al., PNAS 2023, Eq. 4). Unlike ``CVDist``, ``Tq`` does not weight by O-O distance, so it
    is a *count*-like order parameter: ``Tq ~= 2`` whenever exactly one H3O+/OH- pair exists
    anywhere in the box (regardless of how far apart the two ions are), plus a small
    ``N_O * sqrt(c)`` baseline from the intact-water oxygens; ``Tq ~= 4`` for two independent
    pairs, and so on. See :func:`tq_value` and the module docstring for the shared ``q_o``
    definition.

    Typical use (memo: SALT global-CV runs on CVDist/CVNum) is as an *auxiliary* CV that is
    never itself gridded/biased/estimated by an ABF-family method, only restrained with a
    one-sided (upper) harmonic wall placed just above 2, to keep a global-CV run in the
    "exactly one ion pair" sector. See ``pysages.methods.aux_restraint.AuxiliaryWall`` for how
    to attach such a wall to any `SamplingMethod` without adding a grid dimension for ``Tq``.

    Parameters
    ----------
    indices: [oxygen_indices, hydrogen_indices]
    cell: array-like, shape (3, 3) or (3,)
    lam: float, default 8.0 (1/Å)
        Softmax sharpness of each hydrogen's assignment over oxygens (shared with ``CVDist``).
    c: float, default 0.01
        Regularizer under the square root (see :func:`tq_value`).
    """

    def __init__(self, indices, cell, lam=DEFAULT_LAMBDA, c=DEFAULT_TQ_C):
        super().__init__(indices, cell)
        self.lam = float(lam)
        self.c = float(c)

    @property
    def function(self):
        cell, lam, c = self.cell, self.lam, self.c
        return lambda p1, p2: tq_value(p1, p2, cell, lam, c)
