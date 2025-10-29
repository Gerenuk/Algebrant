import math
import numbers
from collections.abc import Callable, Iterable
from types import NotImplementedType
from typing import Any, Self

import numpy as np
from algebrant.algebra.algebra_data import AlgebraData, algebra_mul
from algebrant.algebra.basis import BasisProtocol

# from algebrant.algebra.operations import (
#     algebra_add,
#     algebra_mul,
#     algebra_mul_factor,
# )
from ..common import is_identity
from ..display_config import MAX_ONE_LINE_ELEM
from ..repr_printer import PlainReprMixin

"""
Module implemented with a basis to be able to compare elements

* basis: needs to be hashable; sortable for __repr__
* factor: needs to compare to 0, __neg__, 1/,

Note:
* np.matrix is not able to multiply element-wise inside
"""


ZERO_FLOAT_TOLERANCE = 1e-9


def dot_product(a, b) -> Any:  # TODO: complex return?
    if isinstance(a, Algebra) and isinstance(b, Algebra):  # TODO: `b` needs to be Algebra?
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


def _first_factor(expr):
    if isinstance(expr, Algebra):
        if not expr.basis_factor:
            return 0

        _expr_first_basis, expr_first_factor = sorted(
            expr.basis_factor.basis_factor.items(), key=lambda b_f: b_f[0].sort_key
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
        return float(val.real) < 0 or (val.real == 0 and float(val.imag) < 0)  # TODO: why float()

    if isinstance(val, Algebra):
        return _is_negative(_first_factor(val))

    raise ValueError(f"Unknown type {type(val)} for is_negative")


Factor = Any  # TODO


class Algebra[Basis: BasisProtocol](PlainReprMixin):
    """
    Algebra with a basis

    Always make sure that any basis respects and uses UNITY_BASIS (i.e. return it or overwrite _unity)
    this is because UNITY_BASIS will be created if you add Module + Factor

    Zero is never included in the data
    Will drop terms which are very small compared to the total magnitude
    """

    def __init__(
        self,
        basis_factor: AlgebraData[Basis],
        *,
        basis_class,
        op_prio=1,
    ) -> None:
        """
        op_prio: low is highest prio; less prio will pass operation to more prio
        """
        for key, _val in basis_factor:
            if not isinstance(key, basis_class):  # TODO: subclass ok?
                raise ValueError(
                    f"One basis class in basis_factor of {self.__class__.__name__} is {key.__class__}, but should be {basis_class}"
                )

        self.basis_factor = basis_factor
        self.basis_class = basis_class
        self.unity_basis = basis_class.unity()
        self.op_prio = op_prio

    def _new(self, basis_factor: AlgebraData[Basis]) -> Self:
        """
        used to create results with appropriate initialization of the same properties
        """
        return self.__class__(  # TODO: generalize?
            basis_factor=basis_factor,
            basis_class=self.basis_class,
            op_prio=self.op_prio,
        )

    def _factor_algebradata(self, factor: Factor) -> AlgebraData[Basis]:
        return AlgebraData.make_single(self.unity_basis, factor)

    def to_list(self, levels=None) -> list[tuple]:
        return [
            (basis,) + expr_factor
            for basis, factor in self.basis_factor
            for expr_factor in (
                factor.to_list(levels=(levels - 1) if levels is not None else None)
                if (levels is None or levels > 1) and isinstance(factor, Algebra)
                else [(factor,)]
            )
        ]

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

        return self._new(self.basis_factor + other_wrapped.basis_factor)

    def __radd__(self, first: Any) -> Self:
        return self + first

    def __neg__(self) -> Self:
        return self._new(-self.basis_factor)

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __rsub__(self, first) -> Self:
        return first + (-self)

    def __eq__(self, other) -> bool:  # TODO: check
        if other == 0:
            return all(
                math.isclose(factor, 0, abs_tol=ZERO_FLOAT_TOLERANCE)
                if isinstance(factor, numbers.Real)
                else factor == 0
                for _basis, factor in self.basis_factor
            )

        other = self._ensure_prio(other)

        if other is NotImplemented:
            return NotImplemented

        return self.basis_factor == other.basis_factor

    def __hash__(self) -> int:
        return hash(frozenset(self.basis_factor))

    def _unity(self, factor: Factor) -> Self:
        """
        Returns 1 element
        Needed update upgrade factors with a basis for operations
        e.g. when adding a scalar value
        """
        return self._new(AlgebraData.make_single(self.unity_basis, factor))

    def _zero(self) -> Self:  # rarely needed
        return self._new(AlgebraData({}))

    def _repr_pretty_(self, printer, cycle: bool, support_newlines=True):
        if cycle:
            return printer.text("...")

        if not self.basis_factor:
            printer.text("0")
            return

        multi_line = (
            support_newlines
            and len(self.basis_factor) > 1
            and _is_long_repr(self.basis_factor.to_dict)
        )

        for i, (basis, factor) in enumerate(
            sorted(self.basis_factor, key=lambda b_f: b_f[0].sort_key)
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
                    printer.begin_group(2, "")  # TODO: ?
                else:
                    printer.end_group(2, "")
                    printer.break_()
                    printer.text("+ " if not neg_sign else "- ")
                    printer.begin_group(2, "")
            else:
                if is_first_element:
                    printer.text("" if not neg_sign else "-")
                else:
                    printer.text(" + " if not neg_sign else " - ")

            do_print_factor = basis.is_unity or not is_identity(factor)

            factor_needs_parenthesis = do_print_factor and _repr_needs_parenthesis(factor)

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
            printer.end_group(2, "")  # TODO?

    # def linear_func(self, func: Callable[[tuple[Basis, Factor]], AlgebraData]) -> Self:
    #     # TODO: needed?
    #     return self._new(dict(algebra_map(self.basis_factor.items(), func)))

    # def basis_factor_pairs(self, other: Self) -> dict[Basis, tuple[Factor, Factor]]:
    #     return {
    #         basis: (self.basis_factor[basis], other.basis_factor[basis])
    #         for basis in self.basis_factor.keys() & other.basis_factor.keys()
    #     }

    # def map_basis(self, func: Callable[[Basis], AlgebraData]) -> Self:
    #     return self._new(dict(algebra_map_basis(self.basis_factor.items(), func)))

    # def map_factor(self, func: Callable[[Factor], AlgebraData]) -> Self:
    #     return self._new(dict(algebra_map_factor(self.basis_factor.items(), func)))

    # def map_basisfactor(self, func: Callable[[BasisFactor], AlgebraData]) -> Self:
    #     return self._new(dict(algebra_maps_basisfactor(self.basis_factor.items(), func)))

    def replace(
        self,
        func: Callable[
            [Basis], Iterable[tuple[Basis, Factor]] | None
        ],  # TODO: but should be any BasisProtocol type?
        func_factor: Callable[[Any], Any] | None = None,
        *,
        recursive: bool = True,
    ) -> Self:
        return self._new(self.basis_factor.replace(func, func_factor, recursive=recursive))

    def map(self, func: Callable[[Basis, Factor], Iterable[tuple[Basis, Factor]]]) -> Self:
        return self._new(self.basis_factor.map(func))

    def __mul__(self, other: Any) -> Self | NotImplementedType:
        """
        Implement algebradata.algebra_mul for basis to define multiplication
        """
        return self._mul(other, algebra_mul)

    def __rmul__(self, first: Factor) -> Self:
        """
        other is left-multiplied
        for algebra multiplication implement __mul__
        """
        if first == 1:
            return self

        if first == 0:
            return self._zero()

        # TODO: infinite recursion possible?
        return self._new(self._factor_algebradata(first) * self.basis_factor)

    def _mul(
        self,
        other: Any,
        basis_mul: Callable[[Basis, Factor, Basis, Factor], Iterable[tuple[Basis, Factor]]],
    ) -> Self | NotImplementedType:
        if isinstance(other, numbers.Number):
            if other == 1:
                return self

            if other == 0:
                return self._zero()

        other_wrapped = self._ensure_prio(other)

        if other_wrapped is NotImplemented:
            if isinstance(
                other, self.__class__
            ):  # TODO: ? because same class __radd__ would not be called due to Python
                return other.__rmul__(self)
            return NotImplemented

        try:
            result = self._new(self.basis_factor.mul(other_wrapped.basis_factor, basis_mul))

            return result
        except NotImplementedError:
            raise
            # return NotImplemented

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

    @staticmethod
    def _conjugate_replace(term):
        if hasattr(term, "conjugate"):
            return {term.conjugate(): 1}

        raise NotImplementedError(f"Conjugate not implemented for type {type(term)}")

    @staticmethod
    def _conjugate_replace_factor(factor):
        if hasattr(factor, "conjugate"):
            return factor.conjugate()

        if isinstance(factor, numbers.Real):
            return factor

        raise NotImplementedError(f"Conjugate not implemented for type {type(factor)}")

    def conjugate(self) -> Self:
        """
        Return a new SymbolAlgebra with conjugated symbols.
        """
        return self.replace(self._conjugate_replace, self._conjugate_replace_factor)

    @property
    def c(self) -> Self:
        return self.conjugate()

    def dot(self, other) -> Any:
        """
        assumes all basis are orthonormal
        """
        other_wrapped = self._ensure_prio(other)

        if other_wrapped is NotImplemented:  # TODO: generalize
            raise NotImplementedError(f"Cannot dot {self.__class__} with {other.__class__}")

        return sum(
            dot_product(
                self.basis_factor.basis_factor[key], other_wrapped.basis_factor.basis_factor[key]
            )
            for key in self.basis_factor.basis_factor.keys()
            & other_wrapped.basis_factor.basis_factor.keys()  # TODO
        )

    def __abs__(self) -> Factor:
        """
        topological abs which looks only at coefficents
        """
        return sum(abs(factor) for factor in self.basis_factor.basis_factor.values())

    # def filter(self, func: Callable[[Basis, Factor], bool]) -> Self:
    #     return self._new(
    #         {
    #             basis: factor.filter(func) if isinstance(factor, Algebra) else factor
    #             for basis, factor in self.basis_factor.items()
    #             if func(basis, factor)
    #         }
    #     )


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
        isinstance(factor, Algebra) and _is_long_repr(factor.basis_factor.basis_factor)
        for factor in basis_factor.values()
    )

    return result
