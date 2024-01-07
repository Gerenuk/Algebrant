import random
from itertools import combinations

from .creation import E


def random_vec(*grades, dim, complex=True):
    if not grades:
        grades = range(dim + 1)

    basis = [E(*t) for g in grades for t in combinations(range(1, dim + 1), g)]

    if complex:
        return sum((random.gauss(0, 1) + 1j * random.gauss(0, 1)) * b for b in basis)
    else:
        return sum(random.gauss(0, 1) * b for b in basis)


def random_int_vec(*grades, dim, complex=True, min_int=2, max_int=1000):
    """
    for algebraic tests where you want to avoid floating point imprecision
    """
    if not grades:
        grades = range(dim + 1)

    basis = [E(*t) for g in grades for t in combinations(range(1, dim + 1), g)]

    if complex:
        return sum(
            (
                random.choice([1, -1]) * random.randint(min_int, max_int)
                + 1j * random.choice([1, -1]) * random.randint(min_int, max_int)
            )
            * b
            for b in basis
        )
    else:
        return sum(random.choice([1, -1]) * random.randint(min_int, max_int) * b for b in basis)
