import itertools
from dataclasses import dataclass
from typing import Any, Self

from algebrant.graded.wedge import (
    Wedge,
    WedgeableSortKey,
)
from algebrant.repr_printer import PlainReprMixin
from algebrant.utils import calculated_field


@dataclass(unsafe_hash=True, repr=False)
class WedgeContr(Wedge):
    def lshift(
        self, other: Self, self_factor: Any, other_factor: Any
    ) -> dict[Self, Any]:
        return {
            self.__class__((LeftContraction(contr=self, base=other),)): self_factor
            * other_factor
        }


@dataclass(unsafe_hash=True, repr=False)
class LeftContraction(PlainReprMixin):
    contr: Wedge
    base: Wedge
    grade: int | None = calculated_field()
    is_odd: bool | None = calculated_field()
    wedge_sort_key: WedgeableSortKey = calculated_field()

    def __post_init__(self) -> None:
        contr_grade = self.contr.grade
        base_grade = self.base.grade

        if base_grade is not None and contr_grade is not None:
            self.grade = base_grade - contr_grade
        else:
            self.grade = None

        contr_is_odd = self.contr.is_odd
        base_is_odd = self.base.is_odd
        if contr_is_odd is not None and base_is_odd is not None:
            self.is_odd = contr_is_odd ^ base_is_odd
        else:
            self.is_odd = None

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

        if len(self.contr.elems) >= 2:
            printer.text("(")

        printer.pretty(self.contr)

        if len(self.contr.elems) >= 2:
            printer.text(")")

        printer.text("⌟")

        if len(self.base.elems) >= 2:
            printer.text("(")

        printer.pretty(self.base)

        if len(self.base.elems) >= 2:
            printer.text(")")
