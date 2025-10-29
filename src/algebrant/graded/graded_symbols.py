from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from algebrant.algebra.basis import BasisSortKey
from algebrant.graded.graded_symbol import GradedSymbol
from algebrant.symbols.nc_symbols import NCSymbols
from algebrant.utils import all_not_none


@dataclass(unsafe_hash=True, repr=False)
class GradedSymbols(NCSymbols[GradedSymbol]):
    @property
    def sort_key(self) -> BasisSortKey:
        names = tuple(sym.name for sym in self.symbols)

        grades = self.grades

        if all_not_none(grades):
            return (
                (sum(grades), len(self.symbols)),
                tuple(sorted(names) + list(names)),
            )
        else:
            return ((10000, len(self.symbols)), names)

    @property
    def is_odd(self) -> bool:
        return sum(s.is_odd for s in self.symbols) % 2 == 1

    @property
    def grade(self) -> int | None:
        if len(self.symbols) == 1:
            return self.symbols[0].grade

        return None

    @property
    def grades(self) -> tuple[int | None, ...]:
        return tuple(sym.grade for sym in self.symbols)

    @property
    def _i_sign(self) -> int:
        return -1 if self.is_odd else 1

    @property
    def i(self) -> Iterable[tuple[Self, int]]:
        return [(self, self._i_sign)]

    @property
    def r(self) -> Iterable[tuple[Self, int]]:
        grades = self.grades
        if not all_not_none(grades):
            raise ValueError(f"Cannot do .r_sign due to missing grades {grades} in {self}")

        sign = {1: -1, 0: 1}[sum(grade % 4 in (2, 3) for grade in grades) % 2]
        return [(self._create(tuple(reversed(self.symbols))), sign)]

    @property
    def cl(self) -> Iterable[tuple[Self, int]]:
        grades = self.grades
        if not all_not_none(grades):
            raise ValueError(f"Cannot do .cl_sign due to missing grades {grades} in {self}")

        sign = {1: -1, 0: 1}[sum(grade % 4 in (1, 2) for grade in grades) % 2]
        return [(self._create(tuple(reversed(self.symbols))), sign)]


def i_map(basis: GradedSymbols, factor) -> Iterable[tuple[GradedSymbols, int]]:
    return [(basis, getattr(factor, "i", factor) * basis._i_sign)]
