import dataclasses
from dataclasses import dataclass
from typing import Callable

from algebrant.repr_printer import PlainReprMixin

try:
    import colorful

    colorful.use_true_colors()  # type: ignore
    symbol_col = colorful.limeGreen  # type: ignore
except ImportError:

    def symbol_col(x: str) -> str:
        return x


@dataclass(unsafe_hash=True, order=True, repr=False)
class Symbol(PlainReprMixin):
    name: str
    is_conjugate: bool = False
    color: Callable[[str], str] = symbol_col

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
            return dataclasses.replace(self, is_conjugate=not self.is_conjugate)

        return self
