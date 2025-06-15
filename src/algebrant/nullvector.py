from dataclasses import dataclass
from itertools import groupby
from operator import attrgetter

from .algebra.algebra import Algebra
from .symbols.nc_symbols import NCSymbols
from .symbol import Symbol


@dataclass(unsafe_hash=True, order=True, repr=False)
class NullVector(Symbol):
    def _needs_conjugate(self):
        return True

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            printer.text("...")
            return

        printer.pretty(self.color(self.name + ("†" if self.is_conjugate else "")))


def _reverse_symbols(symbols):
    new_symbols = []

    num_odd = 0

    for _name, sym_gr in groupby(symbols, key=attrgetter("name")):
        sym_gr = tuple(sym_gr)
        new_symbols.extend(sym for sym in reversed(sym_gr))

        if len(sym_gr) % 2 == 1:
            num_odd += 1

    if num_odd % 4 in (2, 3):
        factor_sign = -1
    else:
        factor_sign = 1

    return tuple(new_symbols), factor_sign


class NullVectorAlgebra(Algebra):
    @property
    def r(self):
        """
        Reverse (Anti-involution)
        """
        return self.linear_func("r")

    @property
    def cl(self):
        """
        Clifford conjugate (Involution)
        """
        return self.linear_func("cl")


@dataclass(unsafe_hash=True, order=True, repr=False)
class NullVectorSymbols(NCSymbols):
    """
    includes order normalization
    """

    def __post_init__(self):
        assert _validate_basis_symbols(self.symbols), self.symbols

    def conjugate(self, factor):
        reversed_symbols, factor_sign = _reverse_symbols(self.symbols)

        return self._create(tuple(sym.conjugate() for sym in reversed_symbols)), factor.conjugate() * factor_sign

    def r(self, factor):
        """
        Reverse (Anti-involution)
        """
        reversed_symbols, factor_sign = _reverse_symbols(self.symbols)

        return self._create(reversed_symbols), factor * factor_sign

    def cl(self, factor):
        """
        Clifford conjugate (Involution)
        """
        sign = -1 if len(self.symbols) % 2 == 1 else 1

        return self, sign * factor

    def __mul__(self, other):
        """
        return normalized result
        """

        assert list(self.symbols) == sorted(self.symbols, key=lambda s: s.name), self.symbols
        assert list(other.symbols) == sorted(other.symbols, key=lambda s: s.name), other.symbols

        self_groups = groupby(self.symbols, key=attrgetter("name"))
        other_groups = groupby(other.symbols, key=attrgetter("name"))

        name_ordered = []
        self_name = None
        other_name = None

        remaining_self = len(self.symbols)

        reorder_factor = 1

        # Alphabetic order; merge-sort like
        while 1:
            try:  # get next self
                if self_name is None:
                    self_name, self_symbols = next(self_groups)
                    self_symbols = tuple(self_symbols)
                    remaining_self -= len(self_symbols)

                    assert len(self_symbols) <= 2, self_symbols
            except StopIteration:
                if other_name is not None:
                    name_ordered.extend(other_symbols)

                for _name, gr in other_groups:
                    name_ordered.extend(gr)
                break

            try:  # get next other
                if other_name is None:
                    other_name, other_symbols = next(other_groups)
                    other_symbols = tuple(other_symbols)

                    assert len(other_symbols) <= 2, other_symbols
            except StopIteration:
                if self_name is not None:
                    name_ordered.extend(self_symbols)

                for _name, gr in self_groups:
                    name_ordered.extend(gr)
                break

            #####################
            if self_name < other_name:
                name_ordered.extend(self_symbols)

                self_name = None
            elif other_name < self_name:
                name_ordered.extend(other_symbols)

                other_name = None

                if len(other_symbols) % 2 == 1 and (remaining_self + len(self_symbols)) % 2 == 1:
                    reorder_factor = -reorder_factor
            else:  # equal names
                if self_symbols[-1].is_conjugate == other_symbols[0].is_conjugate:
                    return {}  # *** Return Zero

                both_symbols = self_symbols + other_symbols

                while len(both_symbols) >= 3 and (  # Combine A* A A* or A A* A
                    both_symbols[0].is_conjugate == (not both_symbols[1].is_conjugate) == both_symbols[2].is_conjugate
                ):
                    both_symbols = both_symbols[2:]

                name_ordered.extend(both_symbols)

                self_name = None
                other_name = None

                if len(other_symbols) % 2 == 1 and remaining_self % 2 == 1:
                    reorder_factor = -reorder_factor

        ## Normal order
        # factors = []
        # part = []
        # for _name, group in groupby(name_ordered, key=attrgetter("name")):
        #     group = tuple(group)

        #     assert len(group) <= 2, group

        #     if len(group) == 1 or (len(group) == 2 and group[0].is_conjugate and not group[1].is_conjugate):
        #         part.extend(group)
        #         continue

        #     assert not group[0].is_conjugate and group[1].is_conjugate, group

        #     if part:
        #         factors.append(((part, 1),))
        #         part = []

        #     factors.append(((tuple(), 1), ((group[1], group[0]), -1)))

        # if part:
        #     factors.append(((part, 1),))

        # result = {}
        # for factor_prod in itertools.product(*factors):
        #     basis = self._create(tuple(itertools.chain.from_iterable(basis for basis, _factor in factor_prod)))
        #     total_factor = reorder_factor * math.prod(factor for _basis, factor in factor_prod)

        #     assert basis._validated_basis(), basis.symbols

        #     if basis in result:
        #         result[basis] += total_factor
        #     else:
        #         result[basis] = total_factor

        assert _validate_basis_symbols(name_ordered), name_ordered

        new_basis = self._create(tuple(name_ordered))

        return {new_basis: reorder_factor}


def _validate_basis_symbols(symbols):
    # return list(self.symbols) == sorted(self.symbols, key=lambda s: (s.name, not s.is_conjugate))  # if normal order
    return list(symbols) == sorted(symbols, key=lambda s: s.name)
