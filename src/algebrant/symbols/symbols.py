import dataclasses
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Self

from algebrant.algebra.basis import BasisSortKey
from algebrant.repr_printer import PlainReprMixin
from algebrant.symbols.symbol import Symbol

# SymbolType = TypeVar("SymbolType", bound=Symbol)


@dataclass(repr=False)
class Symbols[SymbolType: Symbol](PlainReprMixin):
    symbol_powers: dict[SymbolType, Any]  # TODO: int?

    def __post_init__(self) -> None:
        if any(x == 0 for x in self.symbol_powers.values()):
            self.symbol_powers = {
                symbol: power for symbol, power in self.symbol_powers.items() if power != 0
            }

    @classmethod
    def unity(cls) -> Self:
        return cls({})

    @property
    def is_unity(self) -> bool:
        return not self.symbol_powers

    def conjugate(self) -> Self:
        return dataclasses.replace(
            self,
            symbol_powers={sym.conjugate(): cnt for sym, cnt in self.symbol_powers.items()},
        )

    def __hash__(self) -> int:
        """
        dict is unhashable
        """
        return hash(frozenset(self.symbol_powers.items()))

    @property
    def sort_key(self) -> BasisSortKey:
        degree = sum(abs(power) for _sym, power in self.symbol_powers.items())
        sorted_symbols = sorted(self.symbol_powers.keys(), key=lambda x: (x.name, x.is_conjugate))

        return (
            (degree, len(self.symbol_powers))
            + tuple(self.symbol_powers[sym] for sym in sorted_symbols),
            tuple(sym.name for sym in sorted_symbols),
        )

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            printer.text("...")
            return

        if not self.symbol_powers:
            printer.text("(1)")
            return

        for i, (symbol, count) in enumerate(
            sorted(self.symbol_powers.items(), key=lambda s_p: s_p[0].name)  # TODO
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
