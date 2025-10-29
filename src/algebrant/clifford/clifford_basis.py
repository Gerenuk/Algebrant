import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from algebrant.algebra.basis import BasisSortKey
from algebrant.clifford.clifford_basis_vec import CliffordBasisVec
from algebrant.repr_printer import PlainReprMixin
from algebrant.utils import calculated_field

"""
Todos (some old?):
* implement conjugate for blades squaring to -1 (needed for .c)
* expand in non-orthogonal basis too
* single element sums unpacked? esp before division
"""


@dataclass(unsafe_hash=True, repr=False)
class CliffordBasis(PlainReprMixin):
    bases: tuple[CliffordBasisVec, ...]  # needs to be sorted (or will be sorted by force)
    sort_key: BasisSortKey = calculated_field()
    is_unity: bool = calculated_field()
    grade: int = calculated_field()
    sqr: int = calculated_field()
    is_odd: bool = calculated_field()

    def __post_init__(self) -> None:
        if list(self.bases) != sorted(self.bases):
            raise ValueError("Bases must be sorted")

        self.sort_key = ((len(self.bases),), tuple(b.name for b in self.bases))
        self.is_unity = self.bases == tuple()
        self.grade = len(self.bases)
        self.sqr = [1, -1][len(self.bases) % 4 in (2, 3)] * math.prod(b.sqr for b in self.bases)
        self.is_odd = len(self.bases) % 2 == 1

    @classmethod
    def unity(cls) -> Self:
        return cls(tuple())

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            return printer.text("...")

        if not self.bases:
            printer.text("(1)")  # should usually not be shown
            return

        for i, basis in enumerate(self.bases):
            if i > 0:
                printer.text(" ")

            printer.pretty(basis)

    def commutes_with(self, other: Self) -> bool:
        len_self = len(self.bases)
        len_other = len(other.bases)
        len_common = len(set(self.bases) & set(other.bases))

        return (len_self * len_other + len_common) % 2 == 0

    @property
    def r(self) -> Iterable[tuple[Self, int]]:
        sign = [1, -1][len(self.bases) % 4 in (2, 3)]

        return [(self, sign)]

    @property
    def i(self) -> Iterable[tuple[Self, int]]:
        sign = [1, -1][len(self.bases) % 2 == 1]

        return [(self, sign)]

    @property
    def cl(self) -> Iterable[tuple[Self, int]]:
        sign = [1, -1][len(self.bases) % 4 in (1, 2)]

        return [(self, sign)]
