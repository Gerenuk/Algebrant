import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Protocol, Self

from algebrant.algebra.basis import BasisSortKey
from algebrant.graded.graded_symbol import GradedSymbol
from algebrant.repr_printer import PlainReprMixin
from algebrant.utils import all_not_none, calculated_field

WedgeableSortKey = tuple[tuple[int, ...], tuple[str, ...]]


if TYPE_CHECKING:
    from IPython.lib.pretty import RepresentationPrinter


class Wedgeable(Protocol):
    @property
    def is_odd(self) -> bool: ...

    @property
    def grade(self) -> int | None: ...

    @property
    def wedge_sort_key(self) -> WedgeableSortKey: ...


def get_valid_grades(elems: Iterable[Wedgeable]) -> Sequence[int] | None:
    grades = [elem.grade for elem in elems]

    if not all_not_none(grades):
        return None

    return grades


@dataclass(unsafe_hash=True, repr=False)
class WedgeableSymbol(PlainReprMixin):
    """
    adds wedge_sort_key to GradedSymbol
    """

    symbol: GradedSymbol
    grade: int | None = calculated_field()
    is_odd: bool = calculated_field()
    wedge_sort_key: WedgeableSortKey = calculated_field()

    def __post_init__(self) -> None:
        self.grade = self.symbol.grade
        self.is_odd = self.symbol.is_odd

        if self.grade is not None:
            self.wedge_sort_key = (
                (
                    -2,
                    self.grade,
                ),
                (self.symbol.name,),
            )
        else:
            self.wedge_sort_key = (
                (
                    -1,
                    0,
                ),
                (self.symbol.name,),
            )

    def _repr_pretty_(self, printer: "RepresentationPrinter", cycle: bool) -> None:
        if cycle:
            printer.text("...")
            return

        printer.pretty(self.symbol)


BasisFactor = tuple["Wedge", Any]


@dataclass(unsafe_hash=True, repr=False)
class Wedge(PlainReprMixin):
    elems: tuple[Wedgeable, ...] = tuple()
    grade: int = calculated_field()
    is_unity: bool = calculated_field()
    sort_key: BasisSortKey = calculated_field()
    is_odd: bool = calculated_field()
    i: Iterable[tuple[Self, int]] = calculated_field()
    r: Iterable[tuple[Self, int]] = calculated_field()
    cl: Iterable[tuple[Self, int]] = calculated_field()

    def __post_init__(self) -> None:
        grades = get_valid_grades(self.elems)
        if grades is not None:
            self.grade = sum(grades)
        else:
            raise NotImplementedError("Wedgeable elements must have a valid grade.")

        self.is_odd = bool(sum(elem.is_odd for elem in self.elems) % 2)

        self.is_unity = not self.elems

        factor_i = -1 if self.is_odd else 1
        factor_r = -1 if self.grade % 4 in (2, 3) else 1

        self.i = [(self, factor_i)]
        self.r = [(self, factor_r)]
        self.cl = [(self, factor_i * factor_r)]

        grade = self.grade
        names = tuple(itertools.chain.from_iterable(elem.wedge_sort_key[1] for elem in self.elems))

        if grade is not None:
            self.sort_key = (
                (0, grade, len(self.elems)),
                names,
            )
        else:
            self.sort_key = ((1, len(self.elems)), names)

    @classmethod
    def xor(
        cls, basis_factor1: BasisFactor, basis_factor2: BasisFactor
    ) -> Iterable[tuple[Self, Any]]:
        basis1, factor1 = basis_factor1
        basis2, factor2 = basis_factor2

        new_wedge = cls(basis1.elems + basis2.elems)
        new_factor = factor1 * factor2

        return [(new_wedge, new_factor)]

    def _repr_pretty_(self, printer: "RepresentationPrinter", cycle: bool) -> None:
        if cycle:
            printer.text("...")
            return

        if not self.elems:
            printer.text("(1)")
            return

        for i, elem in enumerate(self.elems):
            if i > 0:
                printer.text(" ∧ ")

            printer.pretty(elem)

    @classmethod
    def unity(cls) -> Self:
        return cls(tuple())
