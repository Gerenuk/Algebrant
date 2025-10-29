from collections import Counter
from collections.abc import Iterable
from operator import attrgetter
from types import NotImplementedType
from typing import Any, Self, Sequence, Type, TypeVar

import colorful as cf
from algebrant.algebra.algebra_utils import MultiplicationMixin
from algebrant.algebra.algebra_data import AlgebraData, algebra_mul
from algebrant.graded.graded_algebra import GradedAlgebra, commute
from algebrant.graded.graded_symbol import GradedSymbol
from algebrant.operation_prios import WEDGE_CONTR_OP_PRIO
from algebrant.symbols.symbol_algebra import Sym, SymbolAlgebra  # TODO: why needed?
from algebrant.wedge.contraction import LeftContraction, needs_parenthesis
from algebrant.wedge.wedge import Wedge, Wedgeable, WedgeableSortKey, WedgeableSymbol

T = TypeVar("T", bound=Wedge)  # needed since want to create objects of basis_class


@algebra_mul.register
def _(basis1: Wedge, factor1: Any, basis2: Wedge, factor2: Any) -> AlgebraData[Wedge]:
    if basis1.grade == 0:
        return AlgebraData.make_single(basis2, factor1 * factor2)

    result = AlgebraData()

    for basis1, factor2 in commute(basis1, factor2):
        if len(basis1.elems) == 1:
            if basis1.elems[0].grade == 1:
                result += AlgebraData.make_single(Wedge(), factor1 * factor2) * (
                    _vector_contr(Wedge, basis1.elems[0], basis2)
                    + _make_wedge(Wedge, basis1, basis2)
                )
                continue

            else:
                raise NotImplementedError(
                    f"Cannot multiply {basis1} with grade {basis1.elems[0].grade} != 1"
                )

        first_elem = basis1.elems[0]
        if not first_elem.grade == 1:
            raise NotImplementedError(
                f"Cannot multiply {basis1} with first element grade {first_elem.grade} != 1"
            )

        vec = WedgeAlgebra(AlgebraData({Wedge((first_elem,)): 1}))
        remaining = WedgeAlgebra(AlgebraData({Wedge(basis1.elems[1:]): 1}))
        other = WedgeAlgebra(AlgebraData({basis2: 1}))

        basis_result = (vec * (remaining * other) + remaining.i * (vec * other)).basis_factor

        result += AlgebraData.make_single(Wedge(), factor2) * basis_result

    return result.left_mul(0.5 * factor1)


def permutation_parity(perm: Sequence[WedgeableSortKey]) -> int:
    """
    Returns 0 for even parity, 1 for odd parity
    """
    n = len(perm)
    inversions = sum(
        1 for i in range(n) for j in range(i + 1, n) if perm[i] > perm[j]
    )  # TODO: what if equal?
    return inversions % 2


def _parenthesis(elem) -> str:
    if needs_parenthesis(elem):
        return f"({elem})"

    return f"{elem}"


def _dot_symbol(a: Wedge, b: Wedge) -> SymbolAlgebra:
    if a == b and a.grade == 1:
        return Sym(f"{a}²")

    a, b = sorted([a, b], key=attrgetter("sort_key"))

    return Sym(f"{_parenthesis(a)}⋅{_parenthesis(b)}")


def _vector_contr(basis_class: Type[T], contr_vec: Wedgeable, base: Wedge) -> AlgebraData[T]:
    if not contr_vec.grade == 1:
        raise ValueError(f"Expected {contr_vec} to have grade 1, got {contr_vec.grade}")

    result = AlgebraData()

    sign = 1
    for i, elem in enumerate(base.elems):
        if elem.grade == 1:
            result.add(
                basis_class(base.elems[:i] + base.elems[i + 1 :]),
                sign * _dot_symbol(Wedge((contr_vec,)), Wedge((elem,))),
            )
        else:
            result.add(
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


def _make_wedge(basis_class: Type[T], basis1: T, basis2: T) -> AlgebraData[Wedge]:
    sorted_elems, sign = _norm_wedge_sign(basis1.elems + basis2.elems)

    if sign == 0:
        return AlgebraData()

    return AlgebraData.make_single(basis_class(sorted_elems), sign)


def _make_contraction(basis_class: Type[T], contr: T, base: T) -> AlgebraData[Wedge]:
    if contr.grade == 0:
        return AlgebraData.make_single(base, 1)

    if len(base.elems) == 1 and isinstance(base.elems[0], LeftContraction):
        old_contr = contr.elems + base.elems[0].contr.elems
        sorted_contr, sign = _norm_wedge_sign(old_contr)
        if old_contr != sorted_contr:
            return AlgebraData.from_seq(
                (basis, sign * factor)
                for basis, factor in _make_contraction(
                    basis_class, Wedge(sorted_contr), Wedge(base.elems[0].base.elems)
                )
            )

    sign = 1
    for i in reversed(range(len(contr.elems))):
        contr_elem = contr.elems[i]

        if contr_elem.grade == 1:
            expansion = _vector_contr(basis_class, contr_elem, base)

            if len(contr.elems) == 1:
                result = expansion
            else:
                result = AlgebraData.from_seq(expansion).replace(
                    lambda basis: _make_contraction(
                        basis_class,
                        basis_class(contr.elems[:i] + contr.elems[i + 1 :]),
                        basis,
                    ),
                )

            return AlgebraData.from_seq((basis, sign * factor) for basis, factor in result)

        if contr_elem.is_odd:
            sign *= -1

    if contr.grade == base.grade:
        return AlgebraData.make_single(basis_class(), _dot_symbol(contr, base))

    return AlgebraData.make_single(Wedge((LeftContraction(contr, base),)), 1)


# def _make_vec_mul(
#     basis_class: Type[T], basis1: T, basis2: T
# ) -> Iterable[tuple[Wedge, Any]]:
#     if not (len(basis1.elems) != 1 and basis1.elems[0].grade == 1):
#         raise NotImplementedError(
#             f"Cannot vector multiply {basis1} with {len(basis1.elems)} elements and first element grade {basis1.elems[0].grade} != 1"
#         )

#     return algebra_add(
#         _vector_contr(basis_class, basis1.elems[0], basis2),
#         _make_wedge(basis_class, basis1, basis2),
#     )

Factor = Any

WedgeBasisFactor = tuple[Wedge, Factor]
WedgeDictBasisFactor = dict[Wedge, Factor]


class WedgeAlgebra(GradedAlgebra[Wedge], MultiplicationMixin):
    def __init__(
        self,
        basis_factor: AlgebraData[Wedge],
        *,
        basis_class=Wedge,
        op_prio=WEDGE_CONTR_OP_PRIO,
    ) -> None:
        """
        factors must be scalars
        """
        super().__init__(
            basis_factor,
            basis_class=basis_class,
            op_prio=op_prio,
            # normalize_func=_wedge_contr_normalize,
        )

    def _wedge(
        self, basis1: Wedge, factor1: Any, basis2: Wedge, factor2: Any
    ) -> Iterable[tuple[Wedge, Any]]:
        if hasattr(factor1, "vector_conjugate"):
            raise NotImplementedError(
                f"Factor {factor1} has vector_conjugate which is not supported in WedgeAlgebra"
            )

        if hasattr(factor2, "vector_conjugate"):
            raise NotImplementedError(
                f"Factor {factor2} has vector_conjugate which is not supported in WedgeAlgebra"
            )

        return [
            (basis, wedge_factor * factor1 * factor2)
            for basis, wedge_factor in _make_wedge(self.basis_class, basis1, basis2)
        ]

    def __xor__(self, other: Any) -> Self | NotImplementedType:
        return self._mul(other, self._wedge)

    def _lshift(
        self, basis1: Wedge, factor1: Any, basis2: Wedge, factor2: Any
    ) -> Iterable[tuple[Wedge, Any]]:
        if hasattr(factor1, "vector_conjugate"):
            raise NotImplementedError(
                f"Factor {factor1} has vector_conjugate which is not supported in WedgeAlgebra"
            )

        if hasattr(factor2, "vector_conjugate"):
            raise NotImplementedError(
                f"Factor {factor2} has vector_conjugate which is not supported in WedgeAlgebra"
            )

        return [
            (
                basis,
                factor1 * factor2 * factor,
            )
            for basis, factor in _make_contraction(self.basis_class, basis1, basis2)
        ]

    def __lshift__(self, other) -> Self | NotImplementedType:
        return self._mul(other, self._lshift)


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
        AlgebraData({Wedge((_make_mv_sym(name, grade),)): 1}),
    )
