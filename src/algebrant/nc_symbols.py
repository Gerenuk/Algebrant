import dataclasses
import itertools
from dataclasses import dataclass

from ...archive.base_classes import BaseBasis
from .repr_printer import ReprPrinter
from .symbol import BaseSymbol


@dataclass(unsafe_hash=True)  # TODO: not dataclass?
class NCSymbols(BaseBasis):
    symbols: tuple[BaseSymbol]

    def conjugate(self, factor):
        return self._create(
            tuple(sym.conjugate() for sym in reversed(self.symbols))
        ), factor.conjugate()

    @property
    def scalar_part(self):
        raise NotImplementedError()

    @classmethod
    def unity(cls):
        return cls(tuple())

    def _create(self, symbols):
        return dataclasses.replace(self, symbols=symbols)

    def is_unity(self):
        return self.symbols == tuple()

    def _sort_key(self):
        names = tuple(sym.name for sym in self.symbols)

        return (
            len(names),
            tuple(sorted(names)),
            names,
        )

    def __mul__(self, other: "NCSymbols") -> dict:
        # new symbols properties based on left factor
        return {self._create(self.symbols + other.symbols): 1}

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            printer.text("...")
            return

        if not self.symbols:
            printer.text("(1)")
            return

        for i, (symbol, group) in enumerate(itertools.groupby(self.symbols)):
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

    def __repr__(self):
        printer = ReprPrinter()
        self._repr_pretty_(printer, cycle=False)
        return printer.value()
