import itertools
import numbers
import operator
from functools import reduce, singledispatch
from typing import Any, Self

from algebrant.algebra.algebra import BasisProtocol, Factor
from algebrant.algebra.algebra_data import AlgebraData
from algebrant.common import is_zero


@singledispatch
def invert_basis[Basis: BasisProtocol](basis: Basis) -> dict[Basis, Any]:
    raise NotImplementedError(f"inverse not implemented for basis type {type(basis)}")


class MultiplicationMixin:
    # TODO: need to declare that __mul__ and __add__ are required?

    def __pow__(self, power: int) -> Self:
        if not isinstance(power, int):
            raise ValueError(f"Cannot pow by {power}. Only integers implemented.")

        if power == 0:
            return self._unity(1)
        elif power >= 1:
            result = reduce(operator.mul, itertools.repeat(self, power))
        elif power <= -1:
            result = reduce(operator.mul, itertools.repeat(1 / self, abs(power)))
        else:
            raise ValueError(f"Power {power} not recognized")

        return result

    def __truediv__(self, other: Factor) -> Self:
        """
        Only works if invertible
        """
        return self * (1 / other)

    def __rtruediv__(self, numer: Factor) -> Self:
        if numer == 1 and len(self.basis_factor) == 1:
            basis, factor = next(iter(self.basis_factor))

            try:
                inverse_basis = AlgebraData(invert_basis(basis))

                return self._new(inverse_basis) * self._unity(1 / factor)
            except NotImplementedError:
                pass

        return NotImplemented


def clip_small(basis_factor, *, clip_small: float) -> dict[BasisProtocol, Any]:
    return {
        basis: factor
        for basis, factor in basis_factor.items()
        if not is_zero(factor)
        and (
            clip_small is None
            # or max_factor is None
            or not (
                isinstance(factor, numbers.Complex)
                # hasattr(factor, "__abs__")
                and float(abs(factor)) < clip_small  # TODO: ?
            )
        )
    }
