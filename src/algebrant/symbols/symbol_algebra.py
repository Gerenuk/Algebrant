from collections import Counter
from collections.abc import Iterable
from types import NotImplementedType
from typing import Any, Self

from algebrant.algebra.algebra import Algebra
from algebrant.algebra.algebra_utils import MultiplicationMixin
from algebrant.operation_prios import SYMBOL_OP_PRIO
from algebrant.symbols.symbol import Symbol
from algebrant.symbols.symbols import Symbols

BasisFactor = tuple[Symbols, Any]


class SymbolAlgebra(Algebra[Symbols], MultiplicationMixin):
    def _mul(self, bf1: BasisFactor, bf2: BasisFactor) -> Iterable[BasisFactor]:
        new_symbol_powers = Counter(bf1[0].symbol_powers)
        new_symbol_powers.update(bf2[0].symbol_powers)
        new_factor = bf1[1] * bf2[1]

        return [(self.basis_class(dict(new_symbol_powers)), new_factor)]

    def __mul__(self, other) -> Self | NotImplementedType:
        return self._multiply(other, self._mul)


def Sym(name: str, *, power=1) -> SymbolAlgebra:
    """
    Create a wedge from the given elements.
    """
    return SymbolAlgebra(
        {Symbols({Symbol(name): power}): 1},
        basis_class=Symbols,
        op_prio=SYMBOL_OP_PRIO,
    )
