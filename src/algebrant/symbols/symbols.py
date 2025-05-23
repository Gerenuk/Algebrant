import dataclasses
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Self

from algebrant.algebra import BasisSortKey
from algebrant.repr_printer import PlainReprMixin
from algebrant.symbols.symbol import Symbol


def symbol_sort_key(symbol):
    match symbol:
        case Symbol(name, is_conjugate):
            key = (0, name, is_conjugate)
        # case DerivSymbol(symbol, _, derivativates): # TODO: add somewhere without dependencies
        #    key = (1, bool(derivativates), symbol.name)  # TODO: more specific order?
        case _:
            raise ValueError(f"Unknown symbol {symbol} of type {type(symbol)}")

    return key


@dataclass(repr=False)
class Symbols(PlainReprMixin):
    symbol_powers: dict[Symbol, Any]  # TODO: int?

    def __post_init__(self) -> None:
        if any(not isinstance(sym, Symbol) for sym in self.symbol_powers.keys()):
            raise TypeError("All keys must be instances of Symbol")

        if any(x == 0 for x in self.symbol_powers.values()):
            self.symbol_powers = {
                symbol: power
                for symbol, power in self.symbol_powers.items()
                if power != 0
            }

    @classmethod
    def unity(cls) -> Self:
        return cls({})

    @property
    def is_unity(self) -> bool:
        return not self.symbol_powers

    def _new(self, symbol_powers) -> Self:
        return dataclasses.replace(self, symbol_powers=symbol_powers)

    def conjugate(self, factor) -> tuple[Self, Any]:
        return (
            self._new(
                symbol_powers={
                    sym.conjugate(): cnt for sym, cnt in self.symbol_powers.items()
                }
            ),
            factor.conjugate(),
        )

    def __hash__(self) -> int:
        """
        dict is unhashable
        """
        return hash(frozenset(self.symbol_powers.items()))

    def sort_key(self) -> BasisSortKey:
        degree = sum(abs(power) for _sym, power in self.symbol_powers.items())
        symbol_powers = sorted(
            (symbol_sort_key(sym), -power) for sym, power in self.symbol_powers.items()
        )

        return ((degree,), symbol_powers)

    # def inverse(self):
    #     return {
    #         self._new({sym: -power for sym, power in self.symbol_powers.items()}): 1
    #     }

    def mul(self, other: Self, self_factor: Any, other_factor: Any) -> dict:
        new_symbol_powers = Counter(self.symbol_powers)
        new_symbol_powers.update(other.symbol_powers)
        new_factor = self_factor * other_factor

        return {self._new(dict(new_symbol_powers)): new_factor}

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            printer.text("...")
            return

        if not self.symbol_powers:
            printer.text("(1)")
            return

        for i, (symbol, count) in enumerate(
            sorted(self.symbol_powers.items(), key=lambda s_p: symbol_sort_key(s_p[0]))
        ):
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
