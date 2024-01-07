import itertools
from dataclasses import dataclass

from .base_classes import BaseBasis
from .repr_printer import ReprPrinter
from .symbol import BaseSymbol


def default_sort_order(symbols):
    return (
        symbols.__class__.__name__,
        -len(symbols.symbols),
    ) + tuple(sym.name for sym in symbols.symbols)


@dataclass(unsafe_hash=True)
class NCSymbols(BaseBasis):
    symbols: tuple[BaseSymbol]
    sort_order: callable = default_sort_order

    def conjugate(self):
        return NCSymbols(tuple(sym.conjugate() for sym in reversed(self.symbols)))

    def __mul__(self, other: "NCSymbols") -> dict:
        return {NCSymbols(self.symbols + other.symbols): 1}

    def __lt__(self, other):
        return self.sort_order(self) < other.sort_order(other)

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
