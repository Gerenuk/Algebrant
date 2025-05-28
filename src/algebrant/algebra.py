import functools
import itertools
import numbers
from collections.abc import Callable, Iterable, Iterator
from types import NotImplementedType
from typing import Any, Protocol, Self, Type, TypeVar

import numpy as np

from .common import is_identity
from .display_config import MAX_ONE_LINE_ELEM
from .repr_printer import PlainReprMixin

"""
Module implemented with a basis to be able to compare elements

* basis: needs to be hashable; sortable for __repr__
* factor: needs to compare to 0, __neg__, 1/,

Note:
* np.matrix is not able to multiply element-wise inside
"""


def dot_product(a, b) -> Any:  # TODO: complex return?
    if isinstance(a, Algebra) and isinstance(
        b, Algebra
    ):  # TODO: `b` needs to be Algebra?
        return a.dot(b)

    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        return a * b

    if (
        isinstance(a, numbers.Complex)
        and isinstance(b, numbers.Complex)
        and hasattr(a, "conjugate")
    ):
        return a.conjugate() * b

    if (
        isinstance(a, np.ndarray)
        and isinstance(b, np.ndarray)
        and np.issubdtype(a.dtype, np.number)
        and np.issubdtype(b.dtype, np.number)
    ):
        return np.sum(a.conjugate() * b)

    raise ValueError(f"Do not know not how to dot {type(a)} and {type(b)}")


Factor = Any  # TODO


def _first_factor(expr):
    if isinstance(expr, Algebra):
        if not expr.basis_factor:
            return 0

        _expr_first_basis, expr_first_factor = sorted(
            expr.basis_factor.items(), key=lambda b_f: b_f[0].sort_key
        )[0]
        factor = _first_factor(expr_first_factor)

        return factor

    return expr


def _is_negative(val) -> bool:
    """
    used to determine whether to translate "... + -a" into "... - a"
    """
    if isinstance(val, numbers.Real):
        return val < 0

    if isinstance(val, numbers.Complex):
        return float(val.real) < 0 or (
            val.real == 0 and float(val.imag) < 0
        )  # TODO: why float()

    if isinstance(val, Algebra):
        return _is_negative(_first_factor(val))

    raise ValueError(f"Unknown type {type(val)} for is_negative")


BasisSortKey = tuple[tuple[int, ...], tuple[str, ...]]


class BasisProtocol(Protocol):
    """
    basis for algebra with multiplication
    """

    def __hash__(self) -> int: ...

    def __eq__(self, other) -> bool: ...

    @property
    def sort_key(self) -> BasisSortKey: ...

    @property
    def is_unity(self) -> bool: ...


A = TypeVar("A")

BasisFactorPair = tuple[A, Factor]
AlgebraIntermediate = Iterable[BasisFactorPair]


def algebra_add(summands: AlgebraIntermediate) -> AlgebraIntermediate:
    result_dict = {}

    for basis, factor in summands:
        if basis in result_dict:
            result_dict[basis] += factor
        else:
            result_dict[basis] = factor

    return [(b, f) for b, f in result_dict.items() if f != 0]


def algebra_mul(
    *factors: AlgebraIntermediate,
    mul_func: Callable[[BasisFactorPair, BasisFactorPair], AlgebraIntermediate],
) -> AlgebraIntermediate:
    """
    multiply all factors together
    """
    assert len(factors) > 0, "At least one factor is needed for multiplication"

    result = algebra_add(factors[0])

    for factor in factors[1:]:
        new_result_bf = []
        for bf1 in result:
            for bf2 in factor:
                new_result_bf.extend(mul_func(bf1, bf2))

        result = algebra_add(new_result_bf)

    return result


def algebra_map_all_basis(
    summands: AlgebraIntermediate,
    func: Callable[[A], AlgebraIntermediate],
) -> AlgebraIntermediate:
    """
    Apply a function to all summands
    """
    return algebra_add(
        (new_basis, factor * extra_factor)
        for basis, factor in summands
        for new_basis, extra_factor in func(basis)
    )


T = TypeVar("T")


class Algebra[T: BasisProtocol](PlainReprMixin):
    """
    Algebra with a basis

    Always make sure that any basis respects and uses UNITY_BASIS (i.e. return it or overwrite _unity)
    this is because UNITY_BASIS will be created if you add Module + Factor

    Zero is never included in the data
    Will drop terms which are very small compared to the total magnitude
    """

    def __init__(
        self,
        basis_factor: dict[T, Factor],
        *,
        basis_class,
        op_prio=1,
        linear_func_maps=None,
        normalize_func: Callable[[Type[T], T], Iterable[tuple[T, Any]]] | None = None,
    ) -> None:
        """
        op_prio: low is highest prio; less prio will pass operation to more prio
        """
        for key, _val in basis_factor.items():
            if not isinstance(key, basis_class):  # TODO: subclass ok?
                raise ValueError(
                    f"One basis class in basis_factor is {key.__class__}, but should be {basis_class}"
                )

        if normalize_func is not None:
            self.basis_factor: dict[T, Any] = dict(
                algebra_map_all_basis(
                    basis_factor.items(),
                    functools.partial(normalize_func, basis_class),
                )
            )
        else:
            self.basis_factor = basis_factor

        self.basis_class = basis_class
        self.unity_basis = basis_class.unity()
        self.op_prio = op_prio
        self.linear_func_maps = linear_func_maps if linear_func_maps is not None else {}
        self.normalize_func = normalize_func

    def __getattr__(self, attr) -> Any:  # TODO: not Any
        """
        allows to call functions on the basis
        """
        if attr in self.linear_func_maps:
            func = self.linear_func_maps[attr]
            return lambda: self.linear_func(func)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )

    def __mul__(self, other: Any) -> Self | NotImplementedType:
        return self._multiply(other, lambda bf1, bf2: self.basis_class.mul(bf1, bf2))

    def __xor__(self, other: Any) -> Self | NotImplementedType:
        return self._multiply(other, lambda bf1, bf2: self.basis_class.xor(bf1, bf2))

    def __lshift__(self, other: Any) -> Self | NotImplementedType:
        return self._multiply(other, lambda bf1, bf2: self.basis_class.lshift(bf1, bf2))

    def __rshift__(self, other: Any) -> Self | NotImplementedType:
        return self._multiply(other, lambda bf1, bf2: self.basis_class.rshift(bf1, bf2))

    def _new(self, basis_factor: dict[T, Any]) -> Self:
        """
        used to create results with appropriate initialization of the same properties
        """
        return self.__class__(  # TODO: generalize?
            basis_factor=basis_factor,
            basis_class=self.basis_class,
            op_prio=self.op_prio,
            linear_func_maps=self.linear_func_maps,
            normalize_func=self.normalize_func,
        )

    def __iter__(self) -> Iterator[tuple[T, Any]]:
        return iter(self.basis_factor.items())

    def _ensure_prio(self, other: Any) -> Self | NotImplementedType:
        """
        less prio Module are rejected
        more prio
        """
        if isinstance(other, numbers.Number) or (
            hasattr(other, "op_prio") and other.op_prio > self.op_prio
        ):
            return self._unity(other)

        if isinstance(other, self.__class__) and other.op_prio == self.op_prio:
            if self.basis_class == other.basis_class:
                return other
            else:
                raise ValueError(
                    f"Same op_prio={self.op_prio}, but different basis_class {self.basis_class}!={other.basis_class}."
                )

        return NotImplemented

    def __add__(self, other: Any) -> Self:
        if other == 0:
            return self

        other_wrapped = self._ensure_prio(other)

        if other_wrapped is NotImplemented:
            if isinstance(
                other, self.__class__
            ):  # because same class __radd__ would not be called due to Python
                return other.__radd__(self)

            return NotImplemented

        result_basis_factor = dict(
            algebra_add(
                itertools.chain(
                    self.basis_factor.items(), other_wrapped.basis_factor.items()
                )
            )
        )

        return self._new(result_basis_factor)

    def __radd__(self, other) -> Self:
        return self + other

    def __rmul__(self, other: Factor) -> Self:
        """
        other is left-multiplied
        for algebra multiplication implement __mul__
        """
        if other == 1:
            return self

        if other == 0:
            return self._zero()

        basis_factor = {
            basis: other * factor for basis, factor in self.basis_factor.items()
        }

        return self._new(basis_factor)

    def __neg__(self) -> Self:
        return self._new(
            {basis: -factor for basis, factor in self.basis_factor.items()}
        )

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __rsub__(self, first) -> Self:
        return first + (-self)

    def __eq__(self, other) -> bool:
        if other == 0:
            return not self.basis_factor

        other = self._ensure_prio(other)

        if other is NotImplemented:
            return NotImplemented

        if isinstance(other, self.__class__):  # TODO: check?
            return self.basis_factor.keys() == other.basis_factor.keys() and all(
                self.basis_factor[basis] == other.basis_factor[basis]
                for basis in self.basis_factor.keys()
            )

        return self == self._unity(other)

    def __hash__(self) -> int:
        return hash(frozenset(self.basis_factor.items()))

    # def __getitem__(self, basis) -> Factor:
    #     """
    #     for convenient deconstruction
    #     """
    #     return self.basis_factor[basis]

    def _unity(self, factor: Factor) -> Self:
        """
        Returns 1 element
        Needed update upgrade factors with a basis for operations
        e.g. when adding a scalar value
        """
        return self._new({self.unity_basis: factor})

    def _zero(self) -> Self:  # rarely needed
        return self._new({})

    def _repr_pretty_(self, printer, cycle: bool, support_newlines=True):
        if cycle:
            return printer.text("...")

        if not self.basis_factor:
            printer.text("0")
            return

        multi_line = (
            support_newlines
            and len(self.basis_factor) > 1
            and _is_long_repr(self.basis_factor)
        )

        for i, (basis, factor) in enumerate(
            sorted(self.basis_factor.items(), key=lambda b_f: b_f[0].sort_key)
        ):  # assumes basis has __lt__
            is_first_element = i == 0

            if _is_negative(factor):
                neg_sign = True
                factor = -factor
            else:
                neg_sign = False

            if multi_line:
                if is_first_element:
                    printer.text("  " if not neg_sign else " -")
                    printer.begin_group(2)
                else:
                    printer.end_group(2)
                    printer.break_()
                    printer.text("+ " if not neg_sign else "- ")
                    printer.begin_group(2)
            else:
                if is_first_element:
                    printer.text("" if not neg_sign else "-")
                else:
                    printer.text(" + " if not neg_sign else " - ")

            do_print_factor = basis.is_unity or not is_identity(factor)

            factor_needs_parenthesis = do_print_factor and _repr_needs_parenthesis(
                factor
            )

            if factor_needs_parenthesis:
                printer.begin_group(1, "(")

            if do_print_factor:
                printer.pretty(factor)

            if factor_needs_parenthesis:
                printer.end_group(1, ")")

            if not basis.is_unity:
                if do_print_factor:
                    printer.text(" ")

                printer.pretty(basis)

        if multi_line:
            printer.end_group(2)

    def linear_func(self, func) -> Self:
        return self._new(
            dict(
                algebra_add(
                    func(basis, factor) for basis, factor in self.basis_factor.items()
                )
            )
        )

    def _multiply(self, other, basis_mul) -> Self | NotImplementedType:
        other_wrapped = self._ensure_prio(other)

        if other_wrapped is NotImplemented:
            if isinstance(
                other, self.__class__
            ):  # because same class __radd__ would not be called due to Python
                return other.__rmul__(self)
            return NotImplemented

        result_basis_factor = dict(
            algebra_mul(
                self.basis_factor.items(),
                other_wrapped.basis_factor.items(),
                mul_func=basis_mul,
            )
        )

        result = self._new(result_basis_factor)

        return result

    # def __mul__(self, other: Self) -> Self:
    #     """
    #     __rmul__ not needed since Algebra*Algebra
    #     or multiplied from the left by Module

    #     if __mul__ does not support Algebra, then it must be in the factor(?)
    #     TODO: check if this is the case
    #     """
    #     if isinstance(other, numbers.Number):
    #         if other == 1:
    #             return self

    #         if other == 0:
    #             return self._zero()

    #     return self._multiply(other, lambda b1, f1, b2, f2: b1.mul(f1, b2, f2))

    # def __rtruediv__(self, numer) -> Self | Quotient:
    #     """
    #     divide if only 1 element in sum
    #     and returns Quotient otherwise
    #     """
    #     if not self.basis_factor:
    #         raise ZeroDivisionError("Algebra is zero")

    #     # if len(self.basis_factor) == 1:  # TODO: need special rule?
    #     #     basis, factor = next(iter(self.basis_factor.items()))
    #     #     if hasattr(basis, "inverse") and (
    #     #         (inverse := basis.inverse()) is not NotImplemented
    #     #     ):  # in those cases there is a simplified inverse
    #     #         return numer * self._clone(inverse) * (1 / factor)

    #     return Quotient(numer, self)

    def dot(self, other) -> Any:
        """
        assumes all basis are orthonormal
        """
        other_wrapped = self._ensure_prio(other)

        if other_wrapped is NotImplemented:  # TODO: generalize
            raise NotImplementedError(
                f"Cannot dot {self.__class__} with {other.__class__}"
            )

        return sum(
            dot_product(self.basis_factor[key], other_wrapped.basis_factor[key])
            for key in self.basis_factor.keys() & other_wrapped.basis_factor.keys()
        )


def _repr_needs_parenthesis(factor) -> bool:
    # if isinstance(factor, numbers.Complex):
    #    return factor.real != 0 and factor.imag != 0

    if isinstance(factor, Algebra):
        return len(factor.basis_factor) > 1  # or (
        #    len(factor.basis_factor) == 1 and _repr_needs_parenthesis(next(iter(factor.basis_factor.values())))
        # )

    return False


def _is_long_repr(basis_factor: dict) -> bool:
    result = len(basis_factor) > MAX_ONE_LINE_ELEM or any(
        isinstance(factor, Algebra) and _is_long_repr(factor.basis_factor)
        for factor in basis_factor.values()
    )

    return result
