import itertools
import math
from dataclasses import dataclass
from itertools import groupby
from operator import attrgetter

from .nc_symbols import NCSymbols
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


@dataclass(unsafe_hash=True, order=True, repr=False)
class NullVectorSymbols(NCSymbols):
    def __mul__(self, other):
        """
        return normal ordered result
        """

        self_groups = groupby(self.symbols, key=attrgetter("name"))
        other_groups = groupby(other.symbols, key=attrgetter("name"))

        result = []
        self_name = None
        other_name = None

        remaining_self = len(self.symbols)
        factor = 1

        while 1:
            try:  # get next self
                if self_name is None:
                    self_name, self_symbols = next(self_groups)
                    self_symbols = tuple(self_symbols)
                    remaining_self -= len(self_symbols)

                    assert len(self_symbols) <= 2, self_symbols
            except StopIteration:
                # print("Stop self")
                if other_name is not None:
                    result.extend(other_symbols)

                for _name, gr in other_groups:
                    result.extend(gr)
                break

            try:  # get next other
                if other_name is None:
                    other_name, other_symbols = next(other_groups)
                    other_symbols = tuple(other_symbols)

                    assert len(other_symbols) <= 2, other_symbols
            except StopIteration:
                # print("Stop other")
                if self_name is not None:
                    result.extend(self_symbols)

                for _name, gr in self_groups:
                    result.extend(gr)
                break

            # print(f"{self_name=} {self_symbols=} {other_name=} {other_symbols}")

            #####################
            if self_name < other_name:
                result.extend(self_symbols)

                self_name = None
            elif other_name < self_name:
                result.extend(other_symbols)

                other_name = None

                if len(other_symbols) % 2 == 1 and remaining_self % 2 == 1:
                    factor = -factor
            else:  # equal names
                if self_symbols[-1].is_conjugate == other_symbols[0].is_conjugate:
                    return {}  # *** Return Zero

                both_symbols = self_symbols + other_symbols

                # print(f"{both_symbols=}")

                while len(both_symbols) >= 3 and (  # Combine A* A A* or A A* A
                    both_symbols[0].is_conjugate == (not both_symbols[1].is_conjugate) == both_symbols[2].is_conjugate
                ):
                    both_symbols = both_symbols[2:]

                result.extend(both_symbols)

                self_name = None
                other_name = None

                if len(other_symbols) % 2 == 1 and remaining_self % 2 == 1:
                    factor = -factor

        ## Normal order
        factors = []
        part = []
        for _name, group in groupby(result, key=attrgetter("name")):
            group = tuple(group)

            assert len(group) <= 2, group

            if len(group) == 1 or (len(group) == 2 and group[0].is_conjugate and not group[1].is_conjugate):
                part.extend(group)
                continue

            assert not group[0].is_conjugate and group[1].is_conjugate, group

            if part:
                factors.append(((part, 1),))
                part = []

            factors.append(((tuple(), 1), ((group[1], group[0]), -1)))

        if part:
            factors.append(((part, 1),))

        result = {}
        for factor_prod in itertools.product(*factors):
            basis = self._create(tuple(itertools.chain.from_iterable(basis for basis, _factor in factor_prod)))
            total_factor = math.prod(factor for _basis, factor in factor_prod)

            if basis in result:
                result[basis] += total_factor
            else:
                result[basis] = total_factor

        return result
