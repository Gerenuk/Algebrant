from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import singledispatch
from types import NotImplementedType
from typing import Any, Iterator, Self

from algebrant.algebra.basis import BasisProtocol

Factor = Any


@singledispatch
def algebra_mul[Basis](  # TODO: correct?
    basis1: Basis, factor1, basis2: Basis, factor2
) -> Iterable[tuple[Basis, Factor]]:
    # register for singledispatch for each type
    # default for __mul__
    raise NotImplementedError(
        f"algebra_mul not implemented for basis types {type(basis1)} and {type(basis2)}"
    )


@dataclass
class AlgebraData[Basis: BasisProtocol]:
    """
    raw data of the algebra
    does not consider different levels and priorities
    """

    basis_factor: dict[Basis, Factor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.basis_factor = {
            basis: factor for basis, factor in self.basis_factor.items() if factor != 0
        }

    @classmethod
    def from_seq(cls, seq: Iterable[tuple[Basis, Factor]]) -> Self:
        basis_factor: dict[Basis, Factor] = {}

        for basis, factor in seq:
            if basis in basis_factor:
                basis_factor[basis] += factor
            else:
                basis_factor[basis] = factor

        return cls(basis_factor)

    def add(self, basis: Basis, factor: Factor) -> None:
        if basis in self.basis_factor:
            self.basis_factor[basis] += factor
        else:
            self.basis_factor[basis] = factor

        if any(factor == 0 for factor in self.basis_factor.values()):
            self.basis_factor = {
                basis: factor for basis, factor in self.basis_factor.items() if factor != 0
            }

    def __repr__(self) -> str:
        return (
            "{ "
            + " + ".join(f"{factor}⋅{basis}" for basis, factor in self.basis_factor.items())
            + " }"
        )

    def __bool__(self) -> bool:
        return bool(self.basis_factor)

    @property
    def to_dict(self) -> dict[Basis, Factor]:  # TODO: why needed?
        return self.basis_factor.copy()

    @classmethod
    def make_single(cls, basis: Basis, factor: Factor = 1) -> Self:
        return cls({basis: factor})

    def __add__(self, other: Self) -> Self:
        new_basis_factor = self.basis_factor.copy()
        for basis, factor in other.basis_factor.items():
            if basis in new_basis_factor:
                new_basis_factor[basis] += factor
            else:
                new_basis_factor[basis] = factor

        return self.__class__(new_basis_factor)

    def __neg__(self) -> Self:
        return self.__class__({basis: -factor for basis, factor in self.basis_factor.items()})

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def left_mul(self, factor: Factor) -> Self:
        return self.__class__(
            {basis: factor * prev_factor for basis, prev_factor in self.basis_factor.items()}
        )

    def mul(
        self,
        other: Self,
        mul_func: Callable[[Basis, Factor, Basis, Factor], Iterable[tuple[Basis, Factor]]],
    ) -> Self:
        """
        general bilinear multiplication
        """
        result = self.__class__()

        for b1, f1 in self.basis_factor.items():
            for b2, f2 in other.basis_factor.items():
                for new_b, new_f in mul_func(b1, f1, b2, f2):
                    result.add(new_b, new_f)

        return result

    def __mul__(self, other: Self) -> Self | NotImplementedType:
        """
        defaults to algebra_mul
        """
        try:
            return self.mul(other, algebra_mul)
        except NotImplementedError:
            return NotImplemented

    def replace(
        self,
        func: Callable[[Basis], Iterable[tuple[Basis, Factor]] | None],  # TODO: connect types?
        func_factor: Callable[[Any], Any] | None = None,  # TODO: connect types?
        *,
        recursive: bool = True,
    ) -> Self:
        """
        replaces basis and factor separately
        """
        result = self.__class__()

        for basis, factor in self.basis_factor.items():
            new_basis_data = func(basis)

            if recursive and isinstance(factor, AlgebraData):
                factor = factor.replace(func)  # TODO: type correct if Basis different?
            elif func_factor is not None:
                factor = func_factor(factor)

            if new_basis_data is None:
                result.add(basis, factor)
            else:
                for new_basis, new_basis_factor in new_basis_data:
                    result.add(new_basis, factor * new_basis_factor)

        return result

    def map(self, func: Callable[[Basis, Factor], Iterable[tuple[Basis, Factor]]]) -> Self:
        """
        maps a basis-factor pair by a function
        """
        result = self.__class__()

        for basis, factor in self.basis_factor.items():
            for new_basis, new_factor in func(basis, factor):
                result.add(new_basis, new_factor)

        return result

    def __iter__(self) -> Iterator[tuple[Basis, Any]]:
        return iter(self.basis_factor.items())

    def __len__(self) -> int:
        return len(self.basis_factor)
