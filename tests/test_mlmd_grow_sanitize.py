"""``pysages.backends.mlmd._grow_from_first_overflow``: in-step growth must be sized from the
first overflowing frame, never from the chunk's running maximum (2026-10-07: a degenerate
post-overflow frame reported every atom in one leaf and grew the FMM block tile to N)."""
from collections import namedtuple

import numpy as onp
import pytest

pytest.importorskip("mlmd")

from pysages.backends.mlmd import _grow_from_first_overflow  # noqa: E402

Status = namedtuple("Status", "overflow leaf_atom_required")


class FakeCalc:
    def __init__(self):
        self.calls = []
        self._cell = onp.eye(3) * 10.0

    def refresh_device(self, positions, cell=None):
        self.calls.append((onp.asarray(positions).copy(), cell))
        return None, None, Status(overflow=onp.bool_(False), leaf_atom_required=onp.int32(22))


class FakeRefresher:
    def __init__(self, calc):
        self.calc = calc
        self.seen = []

    def grow(self, extras):
        self.seen.append(extras)
        return "grown"


def _extras(first_R, worst_leaf=1536):
    worst = Status(overflow=onp.bool_(True), leaf_atom_required=onp.int32(worst_leaf))
    return ("template", onp.bool_(True), worst, first_R, onp.zeros((3, 3)))


def test_grow_uses_first_overflow_frame_status():
    calc = FakeCalc()
    ref = FakeRefresher(calc)
    R = onp.arange(12.0).reshape(4, 3) + 1.0
    assert _grow_from_first_overflow(ref, _extras(R)) == "grown"
    assert len(calc.calls) == 1 and onp.array_equal(calc.calls[0][0], R)
    assert calc.calls[0][1] is None                       # fixed cell: no cell kwarg
    passed = ref.seen[0]
    assert int(passed[2].leaf_atom_required) == 22        # clean status, not the 1536 maximum
    assert passed[0] == "template" and onp.array_equal(passed[3], R)


def test_falls_back_without_first_frame_or_calc():
    calc = FakeCalc()
    ref = FakeRefresher(calc)
    ex = _extras(onp.zeros((4, 3)))
    _grow_from_first_overflow(ref, ex)
    assert calc.calls == [] and ref.seen[0] is ex          # no recorded frame -> plain grow
    ref2 = FakeRefresher(None)
    ex2 = _extras(onp.ones((4, 3)))
    _grow_from_first_overflow(ref2, ex2)
    assert ref2.seen[0] is ex2                            # no calc -> plain grow
    ref3 = FakeRefresher(calc)
    ex3 = ("t", onp.bool_(True), Status(True, 5))         # old 3-tuple extras -> plain grow
    _grow_from_first_overflow(ref3, ex3)
    assert ref3.seen[0] is ex3


def test_non_finite_first_frame_raises():
    calc = FakeCalc()
    ref = FakeRefresher(calc)
    R = onp.ones((4, 3))
    R[2, 1] = onp.nan
    with pytest.raises(FloatingPointError):
        _grow_from_first_overflow(ref, _extras(R))
    assert calc.calls == [] and ref.seen == []
