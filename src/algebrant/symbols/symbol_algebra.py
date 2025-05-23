from algebrant.algebra import Algebra
from algebrant.symbols.symbol import Symbol
from algebrant.symbols.symbols import Symbols


def Sym(name: str) -> Algebra:
    """
    Create a wedge from the given elements.
    """
    return Algebra(
        {Symbols({Symbol(name): 1}): 1},
        basis_class=Symbols,
        op_prio=1,
    )
