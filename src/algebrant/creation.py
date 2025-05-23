import re
import sys
from itertools import combinations

import colorful

from .algebra import Algebra
from .clifford.clifford import CliffordAlgebra, CliffordBasis
from .derivative.deriv_symbol import DerivSymbol
from .nc_symbols import NCSymbols
from .nullvector import NullVector, NullVectorAlgebra, NullVectorSymbols
from .symbol import Symbol
from .symbols import SymbolAlgebra, Symbols

"""
Main shortcut for creating objects

Most are included with `from algebrant import *` (through `__init__.py`)
"""

CLIFFORD_OP_PRIO = 0
NULLVECTOR_OP_PRIO = 0  # currently no consistent solution with clifford op_prio
NC_SYMBOL_OP_PRIO = 1
SYMBOL_OP_PRIO = 2


NULLVECTOR_COLOR = colorful.hotPink


def sym(name: str, *, power: int = 1):
    return SymbolAlgebra(
        {Symbols({Symbol(name): power}): 1},
        op_prio=SYMBOL_OP_PRIO,
        basis_class=Symbols,
    )


def sym_nc(*names: str):
    if not all(names):
        raise ValueError(f"Empty name in {names}")

    return Algebra(
        {NCSymbols(tuple(Symbol(name) for name in names)): 1},
        op_prio=NC_SYMBOL_OP_PRIO,
        basis_class=NCSymbols,
    )


def E(*bases: str | int):
    """
    will always be deduplicated and sorted
    """

    # def sort(basis):
    #     m = re.search(r"\d+$", basis)
    #     if m:
    #         number = int(m.group(0))
    #     else:
    #         number = float("inf")

    #     return number, basis

    bases = tuple(sys.intern(f"e{b}") if isinstance(b, int) else b for b in set(bases))
    bases = sorted(bases)
    basis = CliffordBasis(tuple(bases))

    return CliffordAlgebra(
        {basis: 1},
        op_prio=CLIFFORD_OP_PRIO,
        basis_class=CliffordBasis,
    )


def Func(name: str, parameters: tuple[str] = tuple(), *, deriv: dict[str, int] = None):
    if deriv is None:
        deriv = {}

    return SymbolAlgebra(
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
        basis_class=Symbols,
    )


def NV(name):
    return NullVectorAlgebra(
        {NullVectorSymbols((NullVector(name, color=NULLVECTOR_COLOR),)): 1},
        op_prio=NULLVECTOR_OP_PRIO,
        basis_class=NullVectorSymbols,
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

    return sum(
        sym(f"{name}{''.join(map(str, t))}") * E(*t)
        for g in grades
        for t in combinations(range(1, dim + 1), r=g)
    )
