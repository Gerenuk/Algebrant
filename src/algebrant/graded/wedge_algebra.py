from collections import Counter
from collections.abc import Iterable
from operator import attrgetter
from typing import Any, Sequence, Type, TypeVar

import colorful as cf

from algebrant.algebra import Algebra
from algebrant.graded.graded_symbol import GradedSymbol
from algebrant.graded.wedge import Wedge, Wedgeable, WedgeableSortKey, WedgeableSymbol
from algebrant.graded.wedge_contr import LeftContraction, WedgeContr

T = TypeVar("T", bound=Wedge)


def MV(name: str, grade: int = 1) -> WedgeableSymbol:
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


def _wedgeable_normalize(basis: T) -> dict[T, int]:
    """
    Normalize the wedgeable basis.
    """
    if isinstance(basis, GradedSymbol):
        return {basis: 1}

    if isinstance(basis, LeftContraction):
        return {basis: 1}

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


def _wedge_normalize(basis: T, basis_class: Type[T]) -> Iterable[tuple[T, Any]]:
    sorted_elems, sign = _norm_wedge_sign(basis.elems)

    if sign == 0:
        return tuple()

    return ((basis_class(tuple(sorted_elems)), sign),)


def _wedge_contr_normalize(basis: T, basis_class: Type[T]) -> Iterable[tuple[T, Any]]:
    return _wedge_normalize(basis, basis_class)


def Wed(*elems: Wedgeable | str) -> Algebra:
    """
    Create a wedge from the given elements.
    """
    new_elems: list[Wedgeable] = [
        MV(elem) if isinstance(elem, str) else elem for elem in elems
    ]

    return Algebra(
        {WedgeContr(tuple(new_elems)): 1},
        basis_class=WedgeContr,
        op_prio=1,
        normalize_func=_wedge_contr_normalize,
    )
