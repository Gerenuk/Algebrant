from collections.abc import Iterable
from typing import Any

from algebrant.graded.graded_symbol_algebra import GradedSymbolAlgebra
from algebrant.graded.graded_symbols import GradedSymbols
from algebrant.wedge.wedge import Wedge, WedgeableSymbol
from algebrant.wedge.wedge_algebra import WedgeAlgebra


def _map_to_wedge(basis: GradedSymbols) -> Iterable[tuple[Wedge, Any]]:
    elems = basis.symbols

    if not elems:
        return [(Wedge(), 1)]

    grades = [elem.grade for elem in elems]
    if not all(grade == 1 for grade in grades):
        raise NotImplementedError(f"Cannot map_to_wedge with grades {grades}")

    result = WedgeAlgebra({Wedge((WedgeableSymbol(elems[0]),)): 1})

    for vec in elems[1:]:
        vec_elem = WedgeAlgebra({Wedge((WedgeableSymbol(vec),)): 1})
        result_i = result.i
        result = -(vec_elem << result_i) + (vec_elem ^ result_i)

    return list(result.basis_factor.items())


def product_to_wedge(expr: GradedSymbolAlgebra) -> WedgeAlgebra:
    return WedgeAlgebra(dict(algebra_map_basis(expr.basis_factor.items(), _map_to_wedge)))
