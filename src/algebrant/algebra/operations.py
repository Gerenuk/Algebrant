import itertools
from collections.abc import Callable
from typing import Any, Iterable, TypeVar

Basis = TypeVar("Basis")
Factor = TypeVar("Factor")

BasisFactor = tuple[Basis, Factor]
AlgebraData = Iterable[BasisFactor]


def algebra_add(*summands: AlgebraData) -> AlgebraData:
    result_dict = {}

    for basis, factor in itertools.chain.from_iterable(summands):
        if basis in result_dict:
            result_dict[basis] += factor
        else:
            result_dict[basis] = factor

    return [(b, f) for b, f in result_dict.items() if f != 0]


def algebra_mul(
    *factors: AlgebraData,
    mul_func: Callable[[BasisFactor, BasisFactor], AlgebraData],
) -> AlgebraData:
    """
    multiply all factors together
    """
    if not factors:
        return tuple()

    result = algebra_add(factors[0])

    for factor in factors[1:]:
        new_result_bf = []
        for bf1 in result:
            for bf2 in factor:
                new_result_bf.extend(mul_func(bf1, bf2))

        result = algebra_add(new_result_bf)

    return result


def algebra_maps_basisfactor(
    basis_factors: AlgebraData, *funcs: Callable[[BasisFactor], AlgebraData]
) -> AlgebraData:
    result = basis_factors

    for func in funcs:
        result = algebra_add(
            (new_basis, new_factor)
            for basis, factor in result
            for new_basis, new_factor in func((basis, factor))
            if new_factor != 0
        )

    return result


def algebra_map_basis(
    summands: AlgebraData,
    func: Callable[[Basis], AlgebraData],
) -> AlgebraData:
    """
    Apply a function to all summands
    """
    return algebra_add(
        (new_basis, factor * extra_factor)
        for basis, factor in summands
        for new_basis, extra_factor in func(basis)
        if factor != 0 and extra_factor != 0
    )


def algebra_map(
    summands: AlgebraData,
    func: Callable[[BasisFactor], AlgebraData],
) -> AlgebraData:
    return algebra_add(*[func(basis_factor) for basis_factor in summands])


def algebra_mul_factor(elem: AlgebraData, factor: Any) -> AlgebraData:
    return [(basis, factor * prev_factor) for basis, prev_factor in elem]
