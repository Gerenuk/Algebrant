from collections import Counter
from collections.abc import Iterable
from typing import Any

from algebrant.algebra.algebra import Algebra
from algebrant.algebra.algebra_utils import MultiplicationMixin, invert_basis
from algebrant.algebra.algebra_data import AlgebraData, algebra_mul
from algebrant.operation_prios import SYMBOL_OP_PRIO
from algebrant.symbols.symbol import Symbol
from algebrant.symbols.symbols import Symbols

BasisFactor = tuple[Symbols, Any]


@algebra_mul.register
def _(
    basis1: Symbols, factor1: Any, basis2: Symbols, factor2: Any
) -> Iterable[tuple[Symbols, Any]]:
    new_symbol_powers = Counter(basis1.symbol_powers)
    new_symbol_powers.update(basis2.symbol_powers)
    new_factor = factor1 * factor2

    return [(Symbols(dict(new_symbol_powers)), new_factor)]


@invert_basis.register
def _(basis: Symbols) -> dict[Symbols, Any]:
    return {Symbols({sym: -power for sym, power in basis.symbol_powers.items()}): 1}


class SymbolAlgebra(Algebra[Symbols[Symbol]], MultiplicationMixin):
    pass


def Sym(name: str, *, power=1, op_prio=SYMBOL_OP_PRIO, **kwargs) -> SymbolAlgebra:
    """
    Create a wedge from the given elements.
    """
    return SymbolAlgebra(
        AlgebraData({Symbols({Symbol(name, **kwargs): power}): 1}),
        basis_class=Symbols,
        op_prio=op_prio,
    )
