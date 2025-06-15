from collections import Counter
from collections.abc import Iterable
from operator import attrgetter
from types import NotImplementedType
from typing import Any, Self, Sequence, Type, TypeVar

import colorful as cf

from algebrant.algebra.algebra import (
    Algebra,
    Factor,
)
from algebrant.algebra.operations import algebra_map_basis
from algebrant.graded.graded_symbol import GradedSymbol
from algebrant.operation_prios import WEDGE_CONTR_OP_PRIO
from algebrant.symbols.symbol_algebra import Sym
from algebrant.wedge.contraction import LeftContraction, needs_parenthesis
from algebrant.wedge.wedge import Wedge, Wedgeable, WedgeableSortKey, WedgeableSymbol

T = TypeVar("T", bound=Wedge)  # needed since want to create objects of basis_class


def permutation_parity(perm: Sequence[WedgeableSortKey]) -> int:
    """
    Returns 0 for even parity, 1 for odd parity
    """
    n = len(perm)
    inversions = sum(
        1 for i in range(n) for j in range(i + 1, n) if perm[i] > perm[j]
    )  # TODO: what if equal?
    return inversions % 2


def _parenthesis(elem):
    if needs_parenthesis(elem):
        return f"({elem})"

    return f"{elem}"


def _dot_symbol(a: Wedge, b: Wedge):
    if a == b and a.grade == 1:
        return Sym(f"{a}", power=2)

    a, b = sorted([a, b], key=attrgetter("sort_key"))

    return Sym(f"{_parenthesis(a)}⋅{_parenthesis(b)}")


def _vector_contr(
    basis_class: Type[T], contr_vec: Wedgeable, base: Wedge
) -> Iterable[tuple[T, Any]]:
    result = []

    sign = 1
    for i, elem in enumerate(base.elems):
        if elem.grade == 1:
            result.append(
                (
                    basis_class(base.elems[:i] + base.elems[i + 1 :]),
                    sign * _dot_symbol(Wedge((contr_vec,)), Wedge((elem,))),
                ),
            )
        else:
            result.append(
                (
                    basis_class(
                        base.elems[:i]
                        + (
                            LeftContraction(
                                basis_class((contr_vec,)),
                                basis_class((elem,)),
                            ),
                        )
                        + base.elems[i + 1 :]
                    ),
                    sign,
                )
            )

        if elem.is_odd:
            sign *= -1

    return result


def _norm_wedge_sign(elems: Iterable[Wedgeable]) -> tuple[tuple[Wedgeable, ...], int]:
    odd_elems = [e for e in elems if e.is_odd]

    counter_odd_elems = Counter(odd_elems)
    for _elem, count in counter_odd_elems.items():
        if count > 1:
            return (tuple(), 0)

    sign = {0: 1, 1: -1}[permutation_parity([e.wedge_sort_key for e in odd_elems])]

    sorted_elems = sorted(elems, key=attrgetter("wedge_sort_key"))

    return tuple(sorted_elems), sign


def _make_contraction(
    basis_class: Type[T], contr: T, base: T
) -> Iterable[tuple[Wedge, Any]]:
    if contr.grade == 0:
        return [(base, 1)]

    if len(base.elems) == 1 and isinstance(base.elems[0], LeftContraction):
        old_contr = contr.elems + base.elems[0].contr.elems
        sorted_contr, sign = _norm_wedge_sign(old_contr)
        if old_contr != sorted_contr:
            return [
                (basis, sign * factor)
                for basis, factor in _make_contraction(
                    basis_class, Wedge(sorted_contr), Wedge(base.elems[0].base.elems)
                )
            ]

    sign = 1
    for i in reversed(range(len(contr.elems))):
        contr_elem = contr.elems[i]

        if contr_elem.grade == 1:
            expansion = _vector_contr(basis_class, contr_elem, base)

            if len(contr.elems) == 1:
                result = expansion
            else:
                result = algebra_map_basis(
                    expansion,
                    lambda basis: _make_contraction(  # recursive call
                        basis_class,
                        basis_class(contr.elems[:i] + contr.elems[i + 1 :]),
                        basis,
                    ),
                )

            return [(basis, sign * factor) for basis, factor in result]

        if contr_elem.is_odd:
            sign *= -1

    if contr.grade == base.grade:
        return [(basis_class(), _dot_symbol(contr, base))]

    return [(Wedge((LeftContraction(contr, base),)), 1)]


WedgeBasisFactor = tuple[Wedge, Factor]
WedgeDictBasisFactor = dict[Wedge, Factor]


class WedgeAlgebra(Algebra[Wedge]):
    def __init__(
        self,
        basis_factor: WedgeDictBasisFactor,
        *,
        basis_class=Wedge,
        op_prio=WEDGE_CONTR_OP_PRIO,
    ) -> None:
        super().__init__(
            basis_factor,
            basis_class=basis_class,
            op_prio=op_prio,
            # normalize_func=_wedge_contr_normalize,
        )

    def _wedge(self, bf1: WedgeBasisFactor, bf2: WedgeBasisFactor):
        basis1, factor1 = bf1
        basis2, factor2 = bf2
        sorted_elems, sign = _norm_wedge_sign(basis1.elems + basis2.elems)
        if sign == 0:
            return tuple()

        return [
            (self.basis_class(sorted_elems), sign * factor1 * factor2)
        ]  # TODO: vector_conjugate?

    def __xor__(self, other: Any) -> Self | NotImplementedType:
        return self._multiply(other, self._wedge)

    def _lshift(
        self, bf1: WedgeBasisFactor, bf2: WedgeBasisFactor
    ) -> Iterable[WedgeBasisFactor]:
        basis1, factor1 = bf1
        basis2, factor2 = bf2

        return [
            (
                basis,
                factor1 * factor2 * factor,
            )
            for basis, factor in _make_contraction(self.basis_class, basis1, basis2)
        ]

    def __lshift__(self, other) -> Self | NotImplementedType:
        return self._multiply(other, self._lshift)

    @property
    def i(self) -> Self:
        return self._new(
            {
                basis: factor * (-1 if basis.is_odd else 1)  # TODO: vector_conjugate?
                for basis, factor in self.basis_factor.items()
            }
        )


def _make_mv_sym(name: str, grade: int = 1) -> WedgeableSymbol:
    return WedgeableSymbol(
        GradedSymbol(
            name,
            base_grade=grade,
            color=cf.yellow if grade == 1 else cf.orange,  # type: ignore
        )
    )


def MVw(name: str, grade: int = 1) -> WedgeAlgebra:
    """
    Create a multivector with the given name and grade.
    """
    return WedgeAlgebra(
        {Wedge((_make_mv_sym(name, grade),)): 1},
    )
