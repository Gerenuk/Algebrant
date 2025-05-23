from itertools import combinations

import numpy as np

from .creation import E


def random_vec(*grades, dim=None, complex=True, int_only=False):
    if not grades and dim is None:
        raise ValueError("Need grades or dim")

    if not grades:
        grades = range(dim + 1)

    if dim is None:
        dim = max(grades)

    bases = [E(*t) for g in grades for t in combinations(range(1, dim + 1), g)]

    return random_from_basis(bases, complex=complex, int_only=int_only)


def random_coef(dim, *, complex=True, int_only=False, min_int=2, max_int=1000):
    match complex, int_only:
        case True, True:
            result = np.random.choice([1, -1], size=dim) * np.random.randint(
                min_int, max_int, size=dim
            ) + 1j * np.random.choice([1, -1], size=dim) * np.random.randint(min_int, max_int, size=dim)

        case True, False:
            result = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)

        case False, True:
            result = np.random.choice([1, -1], size=dim) * np.random.randint(min_int, max_int)

        case False, False:
            result = np.random.normal(size=dim)

    return result


def random_from_basis(bases, *, complex=True, int_only=False, min_int=2, max_int=1000):
    return sum(
        coef * b
        for coef, b in zip(
            bases, random_coef(len(bases), complex=complex, int_only=int_only, min_int=min_int, max_int=max_int)
        )
    )
