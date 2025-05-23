import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self

from algebrant.algebra import BasisSortKey
from algebrant.graded.graded_symbol import GradedSymbol, get_valid_grades
from algebrant.repr_printer import PlainReprMixin
from algebrant.utils import all_not_none, calculated_field

WedgeableSortKey = tuple[tuple[int, ...], tuple[str, ...]]


if TYPE_CHECKING:
    from IPython.lib.pretty import RepresentationPrinter


class Wedgeable(Protocol):
    @property
    def is_odd(self) -> bool | None: ...

    @property
    def grade(self) -> int | None: ...

    @property
    def wedge_sort_key(self) -> WedgeableSortKey: ...


@dataclass(unsafe_hash=True, repr=False)
class WedgeableSymbol(PlainReprMixin):
    """
    adds wedge_sort_key to GradedSymbol
    """

    symbol: GradedSymbol
    grade: int | None = calculated_field()
    is_odd: bool | None = calculated_field()
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


@dataclass(unsafe_hash=True, repr=False)
class Wedge(PlainReprMixin):
    elems: tuple[Wedgeable, ...] = tuple()
    grade: int | None = calculated_field(repr=True)

    def __post_init__(self) -> None:
        grades = get_valid_grades(self.elems)
        if grades is not None:
            self.grade = sum(grades)
        else:
            self.grade = None

        is_odds = [elem.is_odd for elem in self.elems]
        if all_not_none(is_odds):
            self.is_odd = bool(sum(is_odds) % 2)
        else:
            self.is_odd = None

    def xor(self, other: Self, self_factor: Any, other_factor: Any) -> dict[Self, Any]:
        new_wedge = self.__class__(self.elems + other.elems)
        new_factor = self_factor * other_factor

        return {new_wedge: new_factor}

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

    @property
    def is_unity(self) -> bool:
        return not self.elems

    def sort_key(self) -> BasisSortKey:
        """
        Sort key for sorting wedges
        """
        grade = self.grade
        names = tuple(
            itertools.chain.from_iterable(elem.wedge_sort_key[1] for elem in self.elems)
        )

        if grade is not None:
            return (
                (0, grade, len(self.elems)),
                names,
            )
        else:
            return ((1, len(self.elems)), names)
