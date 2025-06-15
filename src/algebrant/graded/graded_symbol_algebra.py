import itertools
import math
from collections.abc import Iterable
from types import NotImplementedType
from typing import Any, Self

from algebrant.algebra.algebra_utils import MultiplicationMixin
from algebrant.algebra.operations import (
    algebra_add,
    algebra_maps_basisfactor,
    algebra_mul,
    algebra_mul_factor,
)
from algebrant.graded.graded_algebra import GradedAlgebra
from algebrant.graded.graded_symbol import GradedSymbol, multivector_color, vector_color
from algebrant.graded.graded_symbols import GradedSymbols
from algebrant.operation_prios import GRADED_SYMBOLS_OP_PRIO
from algebrant.symbols.symbol_algebra import Sym

GradedBasisFactor = tuple[GradedSymbols, Any]


class GradedSymbolAlgebra(GradedAlgebra[GradedSymbols], MultiplicationMixin):
    def _wedge(
        self, bf1: GradedBasisFactor, bf2: GradedBasisFactor
    ) -> Iterable[GradedBasisFactor]:
        basis1, factor1 = bf1
        basis2, factor2 = bf2

        if basis1.grade == 1:
            basis2_inv, sign = basis2.i
            return algebra_mul_factor(
                algebra_add(
                    algebra_mul([bf1], [bf2], mul_func=self._mul),
                    algebra_mul(
                        [(basis2_inv, sign * factor2)], [bf1], mul_func=self._mul
                    ),
                ),
                1 / 2,
            )

        if basis2.grade == 1:
            basis1_inv, sign = basis1.i
            return algebra_mul_factor(
                algebra_add(
                    algebra_mul([bf1], [bf2], mul_func=self._mul),
                    algebra_mul(
                        [bf2], [(basis1_inv, factor1 * sign)], mul_func=self._mul
                    ),
                ),
                1 / 2,
            )

        raise ValueError(
            f"Cannot wedge {basis1} grade {basis1.grade} and {basis2} grade {basis2.grade}"
        )

    def __xor__(self, other: Any) -> Self | NotImplementedType:
        return self._multiply(other, self._wedge)

    def _mul(
        self, bf1: GradedBasisFactor, bf2: GradedBasisFactor
    ) -> Iterable[GradedBasisFactor]:
        basis1, factor1 = bf1
        basis2, factor2 = bf2

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
                self.basis_class(tuple(new_symbols)),
                factor1 * new_factor2 * extra_factor,
            )
        ]

    def __mul__(self, other: Any) -> Self | NotImplementedType:
        return self._multiply(other, self._mul)

    @staticmethod
    def _do_vector_conjugate(basis_factor: GradedBasisFactor) -> GradedBasisFactor:
        basis, factor = basis_factor

        if basis.is_odd and hasattr(factor, "vector_conjugate"):
            # print(f"{basis_factor=} {(basis, factor.vector_conjugate)=}")
            return (basis, factor.vector_conjugate)

        return basis_factor

    @property
    def r(self) -> Self:
        return self.map_basisfactor(
            lambda basis_factor: [
                self._do_vector_conjugate(
                    (
                        basis_r,
                        extra_factor * getattr(basis_factor[1], "r", basis_factor[1]),
                    )
                )
                for basis_r, extra_factor in [basis_factor[0].r]
            ]
        )

    @property
    def cl(self) -> Self:
        return self.map_basisfactor(
            lambda basis_factor: [
                self._do_vector_conjugate(
                    (
                        basis_cl,
                        extra_factor * getattr(basis_factor[1], "cl", basis_factor[1]),
                    )
                )
                for basis_cl, extra_factor in [basis_factor[0].cl]
            ]
        )

    @property
    def i(self) -> Self:
        return self.map_basisfactor(
            lambda basis_factor: [
                (basis_i, extra_factor * getattr(basis_factor[1], "i", basis_factor[1]))
                for basis_i, extra_factor in [basis_factor[0].i]
            ]
        )


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
        {GradedSymbols((_mv_sym(name, grade=grade),)): 1},
        op_prio=GRADED_SYMBOLS_OP_PRIO,
        basis_class=GradedSymbols,
    )
