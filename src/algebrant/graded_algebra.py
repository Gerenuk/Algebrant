import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby

import colorful as cf

from algebrant.algebra import Algebra
from algebrant.nc_symbols import NCSymbols
from algebrant.symbol import Symbol
from algebrant.symbols import SymbolAlgebra, Symbols

vector_color = cf.yellow
multivector_color = cf.orange


@dataclass(unsafe_hash=True, order=True, repr=False)
class GradedSymbol(Symbol):
    grade: int = 0
    power: int = 1

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            printer.text("...")
            return

        printer.pretty(self.color(self.name + ("*" if self.is_conjugate else "")))
        if self.power == 2:
            printer.text("²")
        elif self.power >= 3:
            printer.text(f"^{self.power}")
        elif self.power != 1:
            raise ValueError(f"Unknown {self.power=}")

    # def __repr__(self):
    #    return super().__repr__()

    @property
    def is_odd(self) -> int:
        return (self.grade * self.power) % 2


@dataclass(unsafe_hash=True, repr=False)
class GradedMultivector(NCSymbols):
    @classmethod
    def unity(cls):
        return cls(tuple())

    def _repr_pretty_(self, printer, cycle):
        super()._repr_pretty_(printer, cycle)

    # def __repr__(self):
    #    return super().__repr__()

    @property
    def is_odd(self):
        return sum(s.is_odd for s in self.symbols) % 2 == 1

    def mul(self, factor1, basis2, factor2):
        if self.is_odd and hasattr(factor2, "vector_conjugate"):
            prod_factor = factor1 * factor2.vector_conjugate()
        else:
            prod_factor = factor1 * factor2

        return {basis: prod_factor * extra_factor for basis, extra_factor in (self * basis2).items()}

    def __mul__(self, other):
        combined = self.symbols + other.symbols
        extra_factors = defaultdict(int)

        result_symbols = []
        for key, gr in groupby(combined, key=lambda x: (x.name, x.grade)):
            name, grade = key
            gr = tuple(gr)
            power = sum(x.power for x in gr)

            if grade == 1:
                if power > 1:
                    extra_factors[key] += (power // 2) * 2
                if power % 2 == 1:
                    result_symbols.append(dataclasses.replace(gr[0], power=1))
                continue

            result_symbols.append(dataclasses.replace(gr[0], power=power))

        return {
            self._create(tuple(result_symbols)): (
                SymbolAlgebra(
                    {Symbols(tuple((Symbol(name), power) for (name, _grade), power in extra_factors.items())): 1},
                    op_prio=2,
                    basis_class=Symbols,
                )
                if extra_factors
                else 1
            )
        }


def MV(*names: str, grade=1):
    if not all(names):
        raise ValueError(f"Empty name in {names}")

    return Algebra(
        {
            GradedMultivector(
                tuple(
                    GradedSymbol(
                        name,
                        grade=grade,
                        color={1: vector_color}.get(grade, multivector_color),
                    )
                    for name in names
                ),
            ): 1
        },
        op_prio=0,
        basis_class=GradedMultivector,
    )
