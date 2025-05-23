import dataclasses
import itertools
import numbers
from collections import defaultdict
from dataclasses import dataclass

from algebrant.algebra import Algebra
from algebrant.graded.graded_symbol import GradedSymbol
from algebrant.nc_symbols import NCSymbols


@dataclass(unsafe_hash=True, repr=False)
class GradedMultivector(NCSymbols):
    @classmethod
    def unity(cls):
        return cls(tuple())

    def _repr_pretty_(self, printer, cycle):
        super()._repr_pretty_(printer, cycle)

    # def __repr__(self):
    #    return super().__repr__()

    def _sort_key(self):
        names = tuple(sym.name for sym in self.symbols)

        max_grade = sum(m.max_grade for m in self.symbols)

        return (
            max_grade,
            len(names),
            min([m.max_grade for m in self.symbols], default=0),
            tuple(sorted(names)),
            names,
        )

    @property
    def name(self):
        return "~".join(sym.name for sym in self.symbols)

    @property
    def is_odd(self):
        return sum(s.is_odd for s in self.symbols) % 2 == 1

    @property
    def grade(self):
        if len(self.symbols) == 1:
            return self.symbols[0].grade

        return None

    def mul(self, factor1, basis2, factor2):
        if self.is_odd and hasattr(factor2, "vector_conjugate"):
            prod_factor = factor1 * factor2.vector_conjugate()
        else:
            prod_factor = factor1 * factor2

        return {
            basis: prod_factor * extra_factor
            for basis, extra_factor in (self * basis2).items()
        }

    @staticmethod
    def _comm(basis1, factor1, basis2, factor2, *, sign2):
        grade = basis1.grade

        if grade != 1:
            raise ValueError(f"Cannot comm {basis1} with grade {grade}")

        result = defaultdict(float)

        for basis, factor in basis1.mul(factor1, basis2, factor2).items():
            result[basis] += 1 / 2 * factor

        for basis, factor in basis2.mul(factor2, basis1, factor1).items():
            result[basis] += sign2 / 2 * factor

        return result

    def wedge(self, factor1, basis2, factor2):
        if self.grade == 1:
            sign_involution = {True: -1, False: 1}[basis2.is_odd]
            return self._comm(self, factor1, basis2, factor2, sign2=sign_involution)

        if basis2.grade == 1:
            sign_involution = {True: -1, False: 1}[self.is_odd]
            return self._comm(basis2, factor2, self, factor1, sign2=sign_involution)

        raise ValueError(f"Cannot wedge {self} and {basis2}")

    def left_contr(self, factor1, basis2, factor2):
        if self.grade == 1:
            sign_involution = {True: -1, False: 1}[basis2.is_odd]
            return self._comm(self, factor1, basis2, factor2, sign2=-sign_involution)

        raise ValueError(f"Cannot left_contr {self} and {basis2}")

    def __mul__(self, other):
        combined = self.symbols + other.symbols
        extra_factors = defaultdict(int)

        result_symbols = []  # TODO: into __init__
        for key, gr in itertools.groupby(
            combined, key=lambda x: (x.name, x.grade)
        ):  # TODO: issue with NCSymbols?
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
                    {
                        Symbols(
                            {
                                Symbol(name): power
                                for (name, _grade), power in extra_factors.items()
                            }
                        ): 1
                    },
                    op_prio=2,
                    basis_class=Symbols,
                )
                if extra_factors
                else 1
            )
        }

    @property
    def r(self):
        symbol_factors = [sym.r for sym in self.symbols]
        return self._create(tuple(sym for sym, _ in reversed(symbol_factors))), prod(
            factor for _, factor in symbol_factors
        )


class MVAlgebra(Algebra):
    @property
    def r(self):
        return self._new(
            {
                new_basis: (
                    factor.vector_conjugate()
                    if hasattr(factor, "vector_conjugate") and basis.is_odd
                    else factor
                )
                * extra_factor
                for basis, factor in self.basis_factor.items()
                for new_basis, extra_factor in [basis.r]
            }
        )

    def __xor__(self, other):
        # other_wrapped = self._ensure_prio(other)
        other_wrapped = other  # TODO!

        if other_wrapped is NotImplemented:  # TODO: generalize
            raise NotImplementedError(
                f"Cannot xor {self.__class__} with {other.__class__}"
            )

        if isinstance(other, numbers.Number):
            return self * other

        return self._multiply(other, lambda b1, f1, b2, f2: b1.wedge(f1, b2, f2))

    def __lshift__(self, other):
        other_wrapped = self._ensure_prio(other)

        if other_wrapped is NotImplemented:  # TODO: generalize
            raise NotImplementedError(
                f"Cannot xor {self.__class__} with {other.__class__}"
            )

        if isinstance(other, numbers.Number):
            return self * other

        return self._multiply(other, lambda b1, f1, b2, f2: b1.left_contr(f1, b2, f2))


def MV_sym(name, *, grade=1):
    if not name:
        raise ValueError("Empty name")

    return GradedSymbol(
        name,
        grade=grade,
        color={1: vector_color}.get(grade, multivector_color),
    )  # TODO: confusing?


def MV(*names: str, grade=1) -> MVAlgebra:
    if not all(names):
        raise ValueError(f"Empty name in {names}")

    return MVAlgebra(
        {GradedMultivector(tuple(MV_sym(name, grade=grade) for name in names)): 1},
        op_prio=0,
        basis_class=GradedMultivector,
    )
