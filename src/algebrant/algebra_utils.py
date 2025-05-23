import itertools
import numbers
from abc import abstractmethod
from math import prod
from typing import Any, Self

from algebrant.algebra import BasisProtocol, CumuDict, Factor
from algebrant.common import is_zero


class ArithmeticMixin:
    # TODO: need to declare that __mul__ and __add__ are required?

    def __pow__(self, power: int) -> Self | int:
        if not isinstance(power, int):
            raise ValueError(f"Cannot pow by {power}. Only integers implemented.")

        if power == 0:
            return 1  # assumes 1 is a valid element of the algebra
        elif power >= 1:
            result = prod(itertools.repeat(self, power))
        elif power <= -1:
            result = prod(itertools.repeat(1 / self, abs(power)))
        else:
            raise ValueError(f"Power {power} not recognized")

        return result

    def __truediv__(self, other: Factor) -> Self:
        """
        Only works if invertible
        """
        return self * (1 / other)

    @abstractmethod
    def __rtruediv__(self, numer: Factor) -> Self:
        raise NotImplementedError("__rtruediv__ not implemented")


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
