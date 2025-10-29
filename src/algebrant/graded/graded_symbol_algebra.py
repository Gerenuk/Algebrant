import itertools
import math
from collections.abc import Iterable
from types import NotImplementedType
from typing import Any, Self

from algebrant.algebra.algebra_data import AlgebraData, algebra_mul
from algebrant.algebra.algebra_utils import MultiplicationMixin
from algebrant.graded.graded_algebra import GradedAlgebra
from algebrant.graded.graded_symbol import GradedSymbol, multivector_color, vector_color
from algebrant.graded.graded_symbols import GradedSymbols, i_map
from algebrant.operation_prios import GRADED_SYMBOLS_OP_PRIO
from algebrant.symbols.symbol_algebra import Sym

GradedBasisFactor = tuple[GradedSymbols, Any]


@algebra_mul.register
def _(
    basis1: GradedSymbols, factor1: Any, basis2: GradedSymbols, factor2: Any
) -> Iterable[tuple[GradedSymbols, Any]]:
    prev_symbols = basis1.symbols + basis2.symbols

    new_symbols = []
    extra_factor = 1

    while 1:  # compress vector squares
        new_symbol_factors = []

        for symbol, symbols in itertools.groupby(prev_symbols):
            if symbol.grade != 1:
                new_symbols.extend(symbols)
                continue

            # extract squares of vectors
            num = len(list(symbols))
            if num % 2 == 1:
                new_symbols.append(symbol)

            if num >= 2:
                new_symbol_factors.append(Sym(symbol.name, power=2 * (num // 2)))

        if not new_symbol_factors:
            break

        extra_factor *= math.prod(new_symbol_factors)
        prev_symbols = new_symbols
        new_symbols = []

    if basis1.is_odd and hasattr(factor2, "vector_conjugate"):
        new_factor2 = factor2.vector_conjugate
    else:
        new_factor2 = factor2

    return [
        (
            GradedSymbols(tuple(new_symbols)),
            factor1 * new_factor2 * extra_factor,
        )
    ]


class GradedSymbolAlgebra(GradedAlgebra[GradedSymbols], MultiplicationMixin):
    def _wedge(
        self, basis1: GradedSymbols, factor1: Any, basis2: GradedSymbols, factor2: Any
    ) -> Iterable[GradedBasisFactor]:
        bf1 = AlgebraData.make_single(basis1, factor1)
        bf2 = AlgebraData.make_single(basis2, factor2)

        # TODO: do factors need .i too?

        if basis1.grade == 1:
            return (bf1 * bf2 + bf2.map(i_map) * bf1).left_mul(1 / 2)

        if basis2.grade == 1:
            return (bf1 * bf2 + bf2 * bf1.map(i_map)).left_mul(1 / 2)

        raise ValueError(
            f"Cannot wedge {basis1} grade {basis1.grade} and {basis2} grade {basis2.grade}"
        )

    def __xor__(self, other: Any) -> Self | NotImplementedType:
        return self._mul(other, self._wedge)


def _mv_sym(name, *, grade=1) -> GradedSymbol:
    if not name:
        raise ValueError("Empty name")

    return GradedSymbol(
        name,
        base_grade=grade,
        color={1: vector_color}.get(grade, multivector_color),
    )  # TODO: confusing?


def MV(name: str, grade=1) -> GradedSymbolAlgebra:
    return GradedSymbolAlgebra(
        AlgebraData({GradedSymbols((_mv_sym(name, grade=grade),)): 1}),
        op_prio=GRADED_SYMBOLS_OP_PRIO,
        basis_class=GradedSymbols,
    )
