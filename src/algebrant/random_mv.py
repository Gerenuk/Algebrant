import functools
import operator
from typing import Sequence, TypeVar

import numpy as np


def random_coef(
    dim, *, complex=True, int_only=False, min_int=2, max_int=1000
) -> np.ndarray:
    match complex, int_only:
        case True, True:
            result = np.random.choice([1, -1], size=dim) * np.random.randint(
                min_int, max_int, size=dim
            ) + 1j * np.random.choice([1, -1], size=dim) * np.random.randint(
                min_int, max_int, size=dim
            )

        case True, False:
            result = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)

        case False, True:
            result = np.random.choice([1, -1], size=dim) * np.random.randint(
                min_int, max_int
            )

        case False, False:
            result = np.random.normal(size=dim)

    return result


AlgebraType = TypeVar("AlgebraType")


def random_from_basis(
    bases: Sequence[AlgebraType],
    *,
    complex=True,
    int_only=False,
    min_int=2,
    max_int=1000,
) -> AlgebraType:
    return functools.reduce(
        operator.add,
        [
            coef * b
            for coef, b in zip(
                bases,
                random_coef(
                    len(bases),
                    complex=complex,
                    int_only=int_only,
                    min_int=min_int,
                    max_int=max_int,
                ),
            )
        ],
    )
