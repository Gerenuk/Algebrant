import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Self, TypeVar

from algebrant.algebra.basis import BasisSortKey
from algebrant.repr_printer import PlainReprMixin

Symbol = TypeVar("Symbol")


@dataclass(unsafe_hash=True, repr=False)  # TODO: not dataclass?
class NCSymbols[Symbol](PlainReprMixin):
    symbols: tuple[Symbol, ...] = field(default_factory=tuple)

    @classmethod
    def unity(cls) -> Self:
        return cls(tuple())

    def _create(self, symbols) -> Self:
        return dataclasses.replace(self, symbols=symbols)

    @property
    def is_unity(self) -> bool:
        return self.symbols == tuple()

    @property
    def sort_key(self) -> BasisSortKey:
        return (
            (len(self.symbols),),
            tuple(),
        )

    def _repr_pretty_(self, printer, cycle) -> None:
        if cycle:
            printer.text("...")
            return

        if not self.symbols:
            printer.text("(1)")
            return

        for i, (symbol, group) in enumerate(itertools.groupby(self.symbols)):
            # compress consecutive NCSymbols by power display
            count = len(list(group))

            if i > 0:
                printer.text(" ")

            match count:
                case 1:
                    printer.pretty(symbol)
                case 2:
                    printer.pretty(symbol)
                    printer.text("²")
                case _:
                    printer.pretty(symbol)
                    printer.text("^")
                    printer.pretty(count)
