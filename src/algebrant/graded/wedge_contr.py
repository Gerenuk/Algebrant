import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Self

from algebrant.graded.wedge import (
    Wedge,
    WedgeableSortKey,
)
from algebrant.repr_printer import PlainReprMixin
from algebrant.utils import calculated_field


@dataclass(unsafe_hash=True, repr=False)
class WedgeContr(Wedge):
    @classmethod
    def lshift(
        cls, basis_factor1: tuple[Self, Any], basis_factor2: tuple[Self, Any]
    ) -> Iterable[tuple[Self, Any]]:
        basis1, factor1 = basis_factor1
        basis2, factor2 = basis_factor2
        return [
            (
                cls(
                    (LeftContraction(contr=basis1, base=basis2),),
                ),
                factor1 * factor2,
            )
        ]


@dataclass(unsafe_hash=True, repr=False)
class LeftContraction(PlainReprMixin):
    contr: Wedge
    base: Wedge
    grade: int | None = calculated_field()
    is_odd: bool = calculated_field()
    wedge_sort_key: WedgeableSortKey = calculated_field()

    def __post_init__(self) -> None:
        contr_grade = self.contr.grade
        base_grade = self.base.grade

        if base_grade is not None and contr_grade is not None:
            self.grade = base_grade - contr_grade
        else:
            self.grade = None

        self.is_odd = self.contr.is_odd ^ self.base.is_odd

        names = tuple(
            itertools.chain.from_iterable(
                elem.wedge_sort_key[1]
                for elem in itertools.chain(self.contr.elems, self.base.elems)
            )
        )
        if self.grade is not None:
            self.wedge_sort_key = (
                (
                    0,
                    self.grade,
                ),
                names,
            )
        else:
            self.wedge_sort_key = (
                (
                    0,
                    0,
                ),
                names,
            )

    def _repr_pretty_(self, printer, cycle) -> None:
        if cycle:
            printer.text("...")
            return

        if needs_parenthesis(self.contr):
            printer.text("(")

        printer.pretty(self.contr)

        if needs_parenthesis(self.contr):
            printer.text(")")

        printer.text("⌟")

        if needs_parenthesis(self.base):
            printer.text("(")

        printer.pretty(self.base)

        if needs_parenthesis(self.base):
            printer.text(")")


def needs_parenthesis(elem) -> bool:
    return (
        isinstance(elem, Wedge)
        and (
            len(elem.elems) >= 2
            or any(isinstance(e, LeftContraction) for e in elem.elems)
        )
        or isinstance(elem, LeftContraction)
    )
