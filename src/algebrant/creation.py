import sys
from itertools import combinations

import colorful

from .algebra import Algebra
from .clifford import CliffordAlgebra, CliffordBasis
from .deriv_symbol import DerivSymbol
from .nc_symbols import NCSymbols
from .particle_op import AnnihilationSymbol, particle_normalize, particle_sort_order
from .symbol import Symbol
from .symbols import Symbols

"""
Main shortcut for creating objects

Most are included with `from algebrant import *` (through `__init__.py`)
"""

CLIFFORD_OP_PRIO = 0
PARTICLE_OP_PRIO = 0  # currently particle anc clifford symbols do not interact well
NC_SYMBOL_OP_PRIO = 1
SYMBOL_OP_PRIO = 2


CLIFFORD_UNITY_BASIS = CliffordBasis(tuple())
SYMBOL_UNITY_BASIS = Symbols(frozenset())
NC_SYMBOL_UNITY_BASIS = NCSymbols(tuple())
PARTICLE_UNITY_BASIS = NCSymbols(
    tuple(), sort_order=particle_sort_order
)  # TODO: need other check for unity basis or else __eq__ fails to be precise

PARTICLE_COLOR = colorful.hotPink


def S(name: str, power: int = 1):
    return Algebra(
        {Symbols(frozenset(((Symbol(name), power),))): 1},
        op_prio=SYMBOL_OP_PRIO,
        unity_basis=SYMBOL_UNITY_BASIS,
    )


def Snc(*names: str):
    if not all(names):
        raise ValueError(f"Empty name in {names}")

    return Algebra(
        {NCSymbols(tuple(Symbol(name) for name in names)): 1},
        op_prio=NC_SYMBOL_OP_PRIO,
        unity_basis=NC_SYMBOL_UNITY_BASIS,
    )


def E(*bases: str | int, make_algebra: bool = True):
    """
    will always be deduplicated and sorted
    """
    basis = CliffordBasis(tuple(sorted(sys.intern(f"e{b}") if isinstance(b, int) else b for b in set(bases))))
    if not make_algebra:
        return basis

    return CliffordAlgebra(
        {basis: 1},
        op_prio=CLIFFORD_OP_PRIO,
        unity_basis=CLIFFORD_UNITY_BASIS,
    )


def Func(name: str, parameters: tuple[str] = tuple(), *, deriv: dict[str, int] = None):
    if deriv is None:
        deriv = {}

    return Algebra(
        {
            Symbols(
                (
                    (
                        DerivSymbol(
                            symbol=Symbol(name),
                            parameters=parameters,
                            derivatives=tuple(deriv.items()),
                        ),
                        1,
                    ),
                )
            ): 1
        },
        op_prio=SYMBOL_OP_PRIO,
        unity_basis=SYMBOL_UNITY_BASIS,
    )


def AnnOp(name: str):
    return Algebra(
        {NCSymbols((AnnihilationSymbol(name, color=PARTICLE_COLOR),), sort_order=particle_sort_order): 1},
        op_prio=PARTICLE_OP_PRIO,
        unity_basis=PARTICLE_UNITY_BASIS,
        normalize=particle_normalize,
    )


def make_grades(*grades: int, dim: int = None, name: str = "A"):
    """
    creates symbolic expression with given Clifford grades
    """
    if not grades and dim is None:
        raise ValueError("Need to specify grades and/or dim")

    if not grades:
        grades = range(0, dim + 1)

    if dim is None:
        dim = max(grades)

    return sum(S(f"{name}{''.join(map(str,t))}") * E(*t) for g in grades for t in combinations(range(1, dim + 1), r=g))
