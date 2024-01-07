from dataclasses import dataclass

from .base_classes import BaseSymbol
from .repr_printer import ReprPrinter

try:
    import colorful

    colorful.use_true_colors()
    symbol_col = colorful.limeGreen
except ImportError:
    symbol_col = lambda x: x


@dataclass(unsafe_hash=True, order=True)
class Symbol(BaseSymbol):
    name: str
    is_conjugate: bool = False
    color: callable = symbol_col

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            printer.text("...")
            return

        printer.pretty(self.color(self.name + ("*" if self.is_conjugate else "")))

    def _needs_conjugate(self):
        return self.name and self.name[0].isupper()

    def conjugate(self):
        """
        Only capitalized symbol names are treated as complex
        """
        if self._needs_conjugate():
            return self.__class__(self.name, is_conjugate=not self.is_conjugate, color=self.color)

        return self

    def __repr__(self):
        printer = ReprPrinter()
        self._repr_pretty_(printer, cycle=False)
        return printer.value()
