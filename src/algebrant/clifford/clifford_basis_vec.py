from dataclasses import dataclass
from typing import Any

from algebrant.repr_printer import PlainReprMixin

try:
    import colorful

    colorful.use_true_colors()  # type: ignore
    vec_col = colorful.deepSkyBlue  # type: ignore
    neg_vec_col = colorful.yellowGreen  # type: ignore
except ImportError:

    def vec_col(x):
        return x

    def neg_vec_col(x):
        return x


@dataclass(unsafe_hash=True, order=True, repr=False)
class CliffordBasisVec(PlainReprMixin):
    name: str
    sqr: Any

    def _repr_pretty_(self, printer, cycle) -> None:
        if cycle:
            return printer.text("...")

        if self.sqr == -1:
            col_str = neg_vec_col(self.name)
        else:
            col_str = vec_col(self.name)

        printer.text(str(col_str))
