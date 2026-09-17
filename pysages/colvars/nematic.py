# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Nematic order parameter of a molecular liquid crystal.

S is the standard rank-2 orientational order parameter,

    Q = (3/2) (1/N) sum_i ( u_i u_i^T - I/3 ),      S = eig(Q) of largest |.|

with ``u_i`` the long axis of molecule ``i``.  Q is traceless, so its largest
eigenvalue is never negative; taking the eigenvalue of largest MAGNITUDE is
what makes S span [-1/2, 1], with -1/2 an anti-nematic (oblate) arrangement, 0
isotropic and 1 perfectly aligned.  That is the convention used by Jin et al.,
J. Phys. Chem. A 2024, 128, 7257, who drive S over exactly that range.
``convention="largest"`` selects the largest eigenvalue instead, which is
confined to [0, 1] and is what much of the analysis literature reports.

Two things have to happen before the arithmetic, and both are topology rather
than geometry: molecules must be identified, and a molecule straddling a cell
face must be made whole, since its wrapped coordinates have no meaningful long
axis.  Both are settled once, when the CV is constructed, and reduced to two
small integer tables -- a breadth-first order and a parent index.  What is left
per step is gathers and 3x3 eigendecompositions, which jit and differentiate.

The minimum image is applied to each BOND, never to an atom against the
molecule's centre.  A bond is ~1.5 A and always shorter than half the cell,
whereas a mesogen is comparable to the cell itself: for 5CB or PCH5 in a 20 A
box a centre-relative minimum image folds molecules in half.

Because the CV restores molecular wholeness itself, it does not ask the backend
to unwrap.  It is correct either way: if a backend does supply unwrapped
coordinates the bond vectors are already short and the minimum image is the
identity.

A caution on differentiability.  The gradient of an eigenvalue is singular where
that eigenvalue is degenerate.  For the molecular gyration tensor of a rod-like
mesogen the long axis is well separated, but Q itself approaches degeneracy in a
perfectly isotropic phase (S -> 0), where dS/dr is correspondingly ill
conditioned.  This is a property of the order parameter, not of this
implementation, and it is why biased runs along S are usually started from an
ordered configuration.
"""

import collections

import jax.numpy as np
from jax import lax

from pysages.colvars.core import CollectiveVariable

# A fragment this small cannot define an orientation: a single atom's gyration
# tensor is zero and `eigh` then returns arbitrary eigenvectors, which would
# inject a uniformly random axis into the average.
MIN_FRAGMENT_ATOMS = 5


def _bond_graph(positions, numbers, cell, tolerance):
    """Minimum-image covalent bonds of one frame, as an adjacency list."""
    import numpy as onp
    from ase.data import covalent_radii

    inverse = onp.linalg.inv(cell)
    radii = covalent_radii[numbers]
    delta = positions[None, :, :] - positions[:, None, :]
    delta = delta - onp.round(delta @ inverse) @ cell
    distance = onp.linalg.norm(delta, axis=-1)
    cutoff = tolerance * (radii[:, None] + radii[None, :])
    onp.fill_diagonal(distance, onp.inf)
    adjacency = collections.defaultdict(list)
    for i, j in zip(*onp.nonzero(distance < cutoff)):
        adjacency[int(i)].append(int(j))
    return adjacency


def _components(adjacency, n_atoms):
    import numpy as onp

    seen = onp.zeros(n_atoms, dtype=bool)
    out = []
    for start in range(n_atoms):
        if seen[start]:
            continue
        seen[start] = True
        members, queue = [start], collections.deque([start])
        while queue:
            i = queue.popleft()
            for j in adjacency.get(i, ()):
                if not seen[j]:
                    seen[j] = True
                    members.append(j)
                    queue.append(j)
        out.append(onp.asarray(sorted(members)))
    return out


def _spanning_tree(adjacency, n_atoms):
    """Breadth-first order and parent index, rooted at the molecule's atom 0."""
    import numpy as onp

    order, parent, seen = [], [], {0}
    queue = collections.deque([0])
    while queue:
        i = queue.popleft()
        for j in sorted(adjacency.get(i, ())):
            if j not in seen:
                seen.add(j)
                order.append(j)
                parent.append(i)
                queue.append(j)
    if len(seen) != n_atoms:
        raise ValueError(
            f"the reference molecule is not bonded into one piece: reached "
            f"{len(seen)} of {n_atoms} atoms. Raise `tolerance`."
        )
    return onp.asarray(order, dtype=onp.int32), onp.asarray(parent, dtype=onp.int32)


def nematic_order(positions, order, parent, weights, lengths, n_molecules, largest=False):
    """S for one frame of ``positions`` (all atoms of the molecules, in order).

    ``positions`` is ``[n_molecules * atoms_per_molecule, 3]`` with each
    molecule a contiguous block in the same atom order.
    """
    per = order.shape[0] + 1
    # `asarray` first: `jnp.reshape` hands a numpy input straight back to numpy,
    # and the traced scan index below cannot then be used to slice it.
    r = np.reshape(np.asarray(positions), (n_molecules, per, 3))

    def walk(carry, x):
        j, p = x
        delta = r[:, j, :] - r[:, p, :]
        delta = delta - np.round(delta / lengths) * lengths
        return carry.at[:, j, :].set(carry[:, p, :] + delta), None

    r, _ = lax.scan(walk, r, (order, parent))

    total = np.sum(weights)
    centre = np.einsum("a,mai->mi", weights, r) / total
    centred = r - centre[:, None, :]
    gyration = np.einsum("a,mai,maj->mij", weights, centred, centred) / total
    # `eigh` returns ascending eigenvalues, so the long axis is the last one
    axes = np.linalg.eigh(gyration)[1][:, :, -1]

    tensor = 1.5 * np.einsum("mi,mj->ij", axes, axes) / n_molecules
    tensor = tensor - 0.5 * np.eye(3, dtype=tensor.dtype)
    values = np.linalg.eigvalsh(tensor)
    if largest:
        return values[-1]
    return values[np.argmax(np.abs(values))]


class NematicOrderParameter(CollectiveVariable):
    """
    Nematic order parameter S of a set of identical molecules.

    Construct with :meth:`from_atoms`, which derives the topology from a
    reference frame.  The direct constructor takes the tables instead, so a CV
    can be rebuilt without ASE.

    Parameters
    ----------
    indices: list[int]
        Every atom of every molecule, with each molecule a contiguous block in
        the same atom order.  A tiled molecular cell satisfies this.
    order, parent: array of int
        Breadth-first traversal of one molecule's bonds, and each visited
        atom's parent.  Both have length ``atoms_per_molecule - 1``.
    weights: array of float
        Per-atom weights within a molecule, length ``atoms_per_molecule``.
        Masses reproduce the usual mass-weighted gyration tensor.
    cell: array of float
        The three orthorhombic box lengths.  Fixed: this CV is for NVT.
    n_molecules: int
    convention: {"magnitude", "largest"}
    """

    def __init__(
        self, indices, order, parent, weights, cell, n_molecules, convention="magnitude"
    ):
        import numpy as onp

        super().__init__(indices, None)
        # The CV restores molecular wholeness itself; see the module docstring.
        self.requires_box_unwrapping = False

        if convention not in ("magnitude", "largest"):
            raise ValueError("convention must be 'magnitude' or 'largest'")
        cell = onp.asarray(cell, dtype=float)
        if cell.ndim == 2:
            if not onp.allclose(cell, onp.diag(onp.diag(cell))):
                raise NotImplementedError(
                    "orthorhombic cells only; the minimum image assumes it"
                )
            cell = onp.diag(cell)
        per = len(onp.asarray(weights))
        if len(self.indices) != int(n_molecules) * per:
            raise ValueError(
                f"{len(self.indices)} indices does not match {n_molecules} "
                f"molecules of {per} atoms"
            )

        self.order = np.asarray(onp.asarray(order, dtype=onp.int32))
        self.parent = np.asarray(onp.asarray(parent, dtype=onp.int32))
        self.weights = np.asarray(onp.asarray(weights, dtype=float))
        self.cell = np.asarray(cell)
        self.n_molecules = int(n_molecules)
        self.atoms_per_molecule = per
        self.convention = convention

    @classmethod
    def from_atoms(cls, atoms, tolerance=1.25, mass_weighted=True, convention="magnitude"):
        """Derive the topology from a reference ``ase.Atoms`` frame."""
        import numpy as onp

        positions = onp.asarray(atoms.get_positions(), dtype=float)
        numbers = onp.asarray(atoms.get_atomic_numbers())
        cell = onp.asarray(atoms.get_cell(), dtype=float)
        if not onp.allclose(cell, onp.diag(onp.diag(cell))):
            raise NotImplementedError("orthorhombic cells only")
        n_atoms = len(atoms)

        groups = _components(_bond_graph(positions, numbers, cell, tolerance), n_atoms)
        groups = [g for g in groups if len(g) >= MIN_FRAGMENT_ATOMS]
        sizes = {len(g) for g in groups}
        if len(sizes) != 1:
            raise ValueError(
                f"molecules have different atom counts {sorted(sizes)}; this CV "
                "assumes one repeated species"
            )
        per = sizes.pop()
        for g in groups:
            if not onp.array_equal(g, onp.arange(g[0], g[0] + per)):
                raise ValueError(
                    f"molecule starting at atom {g[0]} is not a contiguous block; "
                    "reorder the frame so each molecule is contiguous"
                )
        starts = onp.asarray(sorted(int(g[0]) for g in groups))
        if not onp.array_equal(starts, onp.arange(len(starts)) * per):
            raise ValueError("molecules do not tile the atom index range")

        order, parent = _spanning_tree(
            _bond_graph(positions[:per], numbers[:per], cell, tolerance), per
        )
        weights = onp.asarray(atoms.get_masses(), dtype=float)[:per]
        if not mass_weighted:
            weights = onp.ones_like(weights)
        indices = list(range(int(starts[0]), int(starts[-1]) + per))
        return cls(
            indices, order, parent, weights, onp.diag(cell), len(groups), convention
        )

    @property
    def function(self):
        """
        Returns
        -------
        Callable
            See `pysages.colvars.nematic.nematic_order` for details.
        """
        order, parent = self.order, self.parent
        weights, cell = self.weights, self.cell
        n_molecules = self.n_molecules
        largest = self.convention == "largest"

        def evaluate(positions):
            return nematic_order(
                positions, order, parent, weights, cell, n_molecules, largest
            )

        return evaluate
