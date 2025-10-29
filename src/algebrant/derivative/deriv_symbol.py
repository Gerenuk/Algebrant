from dataclasses import dataclass

from ....archive.base_classes import BaseSymbol

try:
    import colorful

    colorful.use_true_colors()
    deriv_col = colorful.pink
except ImportError:
    deriv_col = lambda x: x


@dataclass(unsafe_hash=True)
class DerivSymbol(BaseSymbol):
    symbol: BaseSymbol
    parameters: tuple = tuple()
    derivatives: tuple[tuple[str, int]] = tuple()

    def __post_init__(self):
        self.parameters = tuple(sorted(self.parameters))
        self.derivatives = tuple(
            sorted((name, count) for name, count in self.derivatives if count != 0)
        )

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            printer.text("...")
            return

        if self.derivatives:
            printer.pretty(deriv_col("∂/"))
            for deriv, count in self.derivatives:
                printer.pretty(deriv_col(f"∂{deriv}"))
                if count == 1:
                    pass
                elif count == 2:
                    printer.pretty(deriv_col("²"))
                else:
                    printer.pretty(deriv_col(f"^{count}"))
            printer.text("[")
            printer.pretty(self.symbol)
            printer.text("(")
            for i, param in enumerate(self.parameters):
                if i > 0:
                    printer.text(",")
                printer.text(param)
            printer.text(")")
            printer.text("]")
        else:
            printer.pretty(self.symbol)
            printer.text("(")
            for i, param in enumerate(self.parameters):
                if i > 0:
                    printer.text(",")
                printer.text(param)
            printer.text(")")

    def __repr__(self):
        result = [
            "D" if self.derivatives else "",
            ",".join(
                f"{param}" + (f"{order}" if order != 1 else "") for param, order in self.derivatives
            ),
            self.symbol.name,
            "(",
            ",".join(f"{param}" for param in self.parameters),
            ")",
        ]

        return "".join(result)

    def conjugate(self):
        return DerivSymbol(
            symbol=self.symbol.conjugate(),
            parameters=self.parameters,
            derivatives=self.derivatives,
        )
