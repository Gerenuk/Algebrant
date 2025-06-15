import dataclasses
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from types import NotImplementedType
from typing import Any, Self

import colorful

from algebrant.algebra.algebra import Algebra
from algebrant.algebra.algebra_utils import MultiplicationMixin
from algebrant.operation_prios import PSEUDONUMBER_OP_PRIO
from algebrant.symbols.symbol import Symbol
from algebrant.symbols.symbols import Symbols
from algebrant.utils import calculated_field

PSEUDONUMBER_COLOR = colorful.plum  # type: ignore


@dataclass(unsafe_hash=True, repr=False)
class PseudoNumber(Symbol):
    is_conjugate: bool = False  # Pseudoscalar is negated
    is_complex_conjugate: bool = False
    grade: int = 0
    rev_is_neg: bool = calculated_field()
    inv_is_neg: bool = calculated_field()

    def __post_init__(self) -> None:
        self.rev_is_neg = self.grade % 4 in (2, 3)
        self.inv_is_neg = self.grade % 2 == 1

    @property
    def vector_conjugate(self) -> tuple[Self, int]:
        if not self.inv_is_neg:
            return dataclasses.replace(self, is_conjugate=not self.is_conjugate), 1
        else:
            return self, 1

    @property
    def r(self) -> tuple[Self, int]:
        return self.vector_conjugate if self.rev_is_neg else (self, 1)

    @property
    def i(self) -> tuple[Self, int]:
        return self.vector_conjugate if self.inv_is_neg else (self, 1)

    @property
    def cl(self) -> tuple[Self, int]:
        return self.vector_conjugate if self.inv_is_neg ^ self.rev_is_neg else (self, 1)

    def _repr_pretty_(self, printer, cycle) -> None:
        if cycle:
            printer.text("...")
            return

        printer.pretty(
            self.color(
                self.name
                + ("~" if self.is_conjugate else "")
                + ("*" if self.is_complex_conjugate else "")
            )
        )


@dataclass(unsafe_hash=True, repr=False)
class PseudoScalar(PseudoNumber):
    sqr: int = 1  # depends on signature

    @property
    def vector_conjugate(self) -> tuple[Self, int]:
        return self, [-1, 1][self.inv_is_neg]

    @property
    def r(self) -> tuple[Self, int]:
        return self, [1, -1][self.rev_is_neg]

    @property
    def i(self) -> tuple[Self, int]:
        return self, [1, -1][self.inv_is_neg]

    @property
    def cl(self) -> tuple[Self, int]:
        return self, [1, -1][self.inv_is_neg ^ self.rev_is_neg]


class PseudoNumberSymbols(Symbols[PseudoNumber]):
    def _map_symbol(self, func) -> tuple[Self, int]:
        extra_factor = 1
        new_symbol_powers = {}
        for symbol, power in self.symbol_powers.items():
            new_symbol, factor = func(symbol)
            new_symbol_powers[new_symbol] = power
            extra_factor *= factor**power

        return self.__class__(new_symbol_powers), extra_factor

    @property
    def vector_conjugate(self) -> tuple[Self, int]:
        return self._map_symbol(lambda x: x.vector_conjugate)

    @property
    def r(self) -> tuple[Self, int]:
        return self._map_symbol(lambda x: x.r)

    @property
    def i(self) -> tuple[Self, int]:
        return self._map_symbol(lambda x: x.i)

    @property
    def cl(self) -> tuple[Self, int]:
        return self._map_symbol(lambda x: x.cl)


BasisFactor = tuple[PseudoNumberSymbols, Any]


class PseudoScalarAlgebra(Algebra[PseudoNumberSymbols], MultiplicationMixin):
    """
    uses PseudoScalar.sqr to reduce symbol_factors
    """

    @property
    def i(self) -> Self:
        return self.map_basis(lambda basis: [basis.i])

    @property
    def r(self) -> Self:
        return self.map_basis(lambda basis: [basis.r])

    @property
    def cl(self) -> Self:
        return self.map_basis(lambda basis: [basis.cl])

    def _mul(self, bf1: BasisFactor, bf2: BasisFactor) -> Iterable[BasisFactor]:
        basis1, factor1 = bf1
        basis2, factor2 = bf2

        new_factor = factor1 * factor2

        new_symbol_powers = Counter(basis1.symbol_powers)
        new_symbol_powers.update(basis2.symbol_powers)

        if not any(
            isinstance(symbol, PseudoScalar) and power == 2
            for symbol, power in new_symbol_powers.items()
        ):
            return [(self.basis_class(dict(new_symbol_powers)), new_factor)]

        new_symbol_powers_2 = {}

        for symbol, power in new_symbol_powers.items():
            if isinstance(symbol, PseudoScalar) and power == 2:
                new_factor *= symbol.sqr
            else:
                new_symbol_powers_2[symbol] = power

        return [
            (
                self.basis_class(new_symbol_powers_2),
                new_factor,
            )
        ]

    def __mul__(self, other) -> Self | NotImplementedType:
        return self._multiply(other, self._mul)

    @property
    def vector_conjugate(self) -> Self:
        return self._new(
            {
                new_basis: factor * extra_factor
                for basis, factor in self.basis_factor.items()
                for new_basis, extra_factor in [basis.vector_conjugate]
            }
        )


def make_I(*, sqr, grade, name="I") -> PseudoScalarAlgebra:
    return PseudoScalarAlgebra(
        {
            PseudoNumberSymbols(
                {
                    PseudoScalar(
                        name=name, sqr=sqr, grade=grade, color=PSEUDONUMBER_COLOR
                    ): 1
                }
            ): 1
        },
        basis_class=PseudoNumberSymbols,
        op_prio=PSEUDONUMBER_OP_PRIO,
    )


def Sym_ps(name) -> PseudoScalarAlgebra:
    return PseudoScalarAlgebra(
        {
            PseudoNumberSymbols(
                {PseudoNumber(name=name, color=PSEUDONUMBER_COLOR): 1}
            ): 1
        },
        basis_class=PseudoNumberSymbols,
        op_prio=PSEUDONUMBER_OP_PRIO,
    )
