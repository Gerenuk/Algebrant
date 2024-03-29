import dataclasses
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

from .algebra import Algebra
from .base_classes import BaseBasis
from .display_config import symbol_sort_key
from .repr_printer import ReprPrinter
from .symbol import BaseSymbol


@dataclass(unsafe_hash=True)
class Symbols(BaseBasis):
    symbol_powers: frozenset[tuple[BaseSymbol, int]]

    def __post_init__(self):
        self.symbol_powers = frozenset((sym, cnt) for sym, cnt in self.symbol_powers if cnt != 0)

    @property
    def scalar_part(self):
        return self

    @classmethod
    def unity(cls):
        return cls(frozenset())

    def is_unity(self):
        return self.symbol_powers == frozenset()

    def _create(self, symbol_powers):
        return dataclasses.replace(self, symbol_powers=symbol_powers)

    def conjugate(self, factor):
        return (
            self._create(symbol_powers=frozenset((sym.conjugate(), cnt) for sym, cnt in self.symbol_powers)),
            factor.conjugate(),
        )

    def _sort_key(self):
        degree = sum(abs(power) for _sym, power in self.symbol_powers)
        symbol_powers = sorted((symbol_sort_key(sym), -power) for sym, power in self.symbol_powers)

        return (degree, symbol_powers)

    def inverse(self):
        return {self._create(frozenset(((sym, -power) for sym, power in self.symbol_powers))): 1}

    def __mul__(self, other: "Symbols") -> dict:
        symbol_powers = Counter(dict(self.symbol_powers))
        symbol_powers.update(dict(other.symbol_powers))  # so that do not lose negative

        return {self._create(frozenset((symbol_powers.items()))): 1}

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            printer.text("...")
            return

        if not self.symbol_powers:
            printer.text("(1)")
            return

        for i, (symbol, count) in enumerate(sorted(self.symbol_powers, key=lambda s_p: symbol_sort_key(s_p[0]))):
            if i > 0:
                printer.text(" ")

            match count:
                case 1:
                    printer.pretty(symbol)
                case 2:
                    printer.pretty(symbol)
                    printer.text("²")
                case Fraction():
                    printer.pretty(symbol)
                    printer.text(f"^({count.numerator}/{count.denominator}")
                case 0.5:
                    printer.text("√")
                    printer.pretty(symbol)
                case -1:
                    printer.pretty(symbol)
                    printer.text("⁻¹")
                case -2:
                    printer.pretty(symbol)
                    printer.text("⁻²")
                # elif count == 1 / 3:
                #    result = f"³√{symbol_text}"
                # elif (1 / count).is_integer():
                #    result = f"{symbol_text}^(1/{int(1/count)})"
                # if count == 3:   # hard to read
                #    return f"{symbol}³"
                case _:
                    printer.pretty(symbol)
                    printer.text("^")
                    printer.pretty(count)

    def __repr__(self):
        printer = ReprPrinter()
        self._repr_pretty_(printer, cycle=False)
        return printer.value()


class SymbolAlgebra(Algebra):
    @property
    def scalar_part(self):
        return self
