import itertools
from collections import Counter
from collections.abc import Iterable
from functools import partial
from operator import attrgetter
from typing import Any, Sequence, Type, TypeVar

import colorful as cf

from algebrant.algebra import Algebra, algebra_add, algebra_map_all_basis, algebra_mul
from algebrant.graded.graded_symbol import GradedSymbol
from algebrant.graded.wedge import Wedge, Wedgeable, WedgeableSortKey, WedgeableSymbol
from algebrant.graded.wedge_contr import LeftContraction, WedgeContr, needs_parenthesis
from algebrant.symbols.symbol_algebra import Sym

T = TypeVar("T", bound=Wedge)  # needed since want to create objects of basis_class


ALGEBRA_WEDGE_CONTR_PRIO = 1


def _make_mv(name: str, grade: int = 1) -> WedgeableSymbol:
    return WedgeableSymbol(
        GradedSymbol(
            name,
            base_grade=grade,
            color=cf.yellow if grade == 1 else cf.orange,  # type: ignore
        )
    )


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


def _dot_symbol(a, b):
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
                    sign * _dot_symbol(contr_vec, elem),
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


def _wedgeable_normalize(
    basis_class: Type[T], basis: Wedgeable
) -> Iterable[tuple[T, Any]]:
    """
    Normalize the wedgeable basis.
    """
    match basis:
        case WedgeableSymbol():
            return [(basis_class((basis,)), 1)]
        case LeftContraction(contr, base):
            sign = 1
            for i in reversed(range(len(contr.elems))):
                contr_elem = contr.elems[i]

                if contr_elem.grade == 1:
                    expansion = _vector_contr(basis_class, contr_elem, base)

                    if len(contr.elems) == 1:
                        result = expansion
                    else:
                        result = algebra_map_all_basis(
                            expansion,
                            lambda basis: _wedgeable_normalize(  # recursive call
                                basis_class,
                                LeftContraction(
                                    basis_class(contr.elems[:i] + contr.elems[i + 1 :]),
                                    basis,
                                ),
                            ),
                        )

                    return [(basis, sign * factor) for basis, factor in result]

                if contr_elem.is_odd:
                    sign *= -1

            if contr.grade == base.grade:
                return [(basis_class(), _dot_symbol(contr, base))]

            return [(basis_class((basis,)), 1)]
        case _:
            raise ValueError(
                f"Cannot normalize basis of type {basis.__class__} with value {basis}"
            )


def _norm_wedge_sign(elems: Iterable[Wedgeable]) -> tuple[Sequence[Wedgeable], int]:
    odd_elems = [e for e in elems if e.is_odd]

    counter_odd_elems = Counter(odd_elems)
    for _elem, count in counter_odd_elems.items():
        if count > 1:
            return (tuple(), 0)

    sign = {0: 1, 1: -1}[permutation_parity([e.wedge_sort_key for e in odd_elems])]

    sorted_elems = sorted(elems, key=attrgetter("wedge_sort_key"))

    return sorted_elems, sign


def _wedge_normalize(basis_class: Type[T], basis: T) -> Iterable[tuple[T, Any]]:
    sorted_elems, sign = _norm_wedge_sign(basis.elems)

    if sign == 0:
        return tuple()

    return ((basis_class(tuple(sorted_elems)), sign),)


def _wedge_contr_normalize(basis_class: Type[T], basis: T) -> Iterable[tuple[T, Any]]:
    wedgeable_norm = algebra_mul(
        *[_wedgeable_normalize(basis_class, elem) for elem in basis.elems],
        mul_func=basis_class.xor,
    )

    # wedgeable_norm = [(basis, 1)]

    wedge_normed = algebra_map_all_basis(
        wedgeable_norm, partial(_wedge_normalize, basis_class)
    )

    result = algebra_add(wedge_normed)

    return result


def make_wedge(*elems: Wedgeable | str) -> Algebra[WedgeContr]:
    """
    Create a wedge from the given elements.
    """
    new_elems: list[Wedgeable] = [
        _make_mv(elem) if isinstance(elem, str) else elem for elem in elems
    ]

    return Algebra(
        {WedgeContr(tuple(new_elems)): 1},
        basis_class=WedgeContr,
        op_prio=ALGEBRA_WEDGE_CONTR_PRIO,
        normalize_func=_wedge_contr_normalize,
    )


def MV(name: str, grade: int = 1) -> Algebra[WedgeContr]:
    """
    Create a multivector with the given name and grade.
    """
    return make_wedge(_make_mv(name, grade))
