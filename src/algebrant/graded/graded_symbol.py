from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, Self

import colorful as cf

from algebrant.repr_printer import PlainReprMixin
from algebrant.symbols.symbol import Symbol
from algebrant.utils import all_not_none, calculated_field

vector_color = cf.yellow  # type: ignore
multivector_color = cf.orange  # type: ignore


class Graded(Protocol):
    @property
    def grade(self) -> int | None: ...


def get_valid_grades(elems: Iterable[Graded]) -> Sequence[int] | None:
    grades = [elem.grade for elem in elems]

    if not all_not_none(grades):
        return None

    return grades


@dataclass(unsafe_hash=True, order=True, repr=False)
class GradedSymbol(Symbol, PlainReprMixin):
    """
    adds power
    """

    base_grade: int = 0
    power: int = 1  # TODO: split into version without power?
    is_odd: bool | None = calculated_field()
    grade: int | None = calculated_field()
    r: tuple[Self, int] = calculated_field()

    def __post_init__(self) -> None:
        self.is_odd = bool((self.base_grade * self.power) % 2)
        self.grade = self.base_grade if self.power == 1 else None
        self.r = (
            (self, -1 if self.power % 2 == 1 else 1)
            if self.base_grade % 4 in (2, 3)
            else (self, 1)
        )

    def _repr_pretty_(self, printer, cycle) -> None:
        if cycle:
            printer.text("...")
            return

        printer.pretty(self.color(self.name + ("*" if self.is_conjugate else "")))
        if self.power == 2:
            printer.text("²")
        elif self.power >= 3:
            printer.text(f"^{self.power}")
        elif self.power != 1:
            raise ValueError(f"Unknown {self.power=}")
