import itertools
import numbers
from math import prod
from typing import Any

from .base_classes import BaseBasis
from .common import is_identity, is_zero
from .display_config import MAX_ONE_LINE_ELEM
from .quotient import Quotient
from .repr_printer import ReprPrinter

"""
Module implemented with a basis to be able to compare elements

* basis: needs to be hashable; sortable for __repr__
* factor: needs to compare to 0, __neg__, 1/,

Note:
* np.matrix is not able to multiply element-wise inside
"""

Factor = Any


def first_factor(expr):
    if isinstance(expr, Algebra):
        if not expr.basis_factor:
            return 0

        _expr_first_basis, expr_first_factor = sorted(expr.basis_factor.items(), key=lambda b_f: b_f[0]._sort_key())[0]
        factor = first_factor(expr_first_factor)

        return factor

    return expr


def is_negative(val):
    """
    used to determine whether to translate "... + -a" into "... - a"
    """
    if isinstance(val, numbers.Real):
        return val < 0

    if isinstance(val, numbers.Complex):
        return val.real < 0 or (val.real == 0 and val.imag < 0)

    if isinstance(val, Algebra):
        return is_negative(first_factor(val))

    raise ValueError(f"Unknown type {type(val)} for is_negative")


class ArithmeticMixin:
    # TODO: need to declare that __mul__ and __add__ are required?

    def __pow__(self, power: int):
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

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, first):
        return first + (-self)

    def __truediv__(self, other: Factor) -> "Module":
        """
        Only works if invertible
        """
        return self * (1 / other)


class Module(ArithmeticMixin):
    """
    Module with a basis

    Always make sure that any basis respects and uses UNITY_BASIS (i.e. return it or overwrite _unity)
    this is because UNITY_BASIS will be created if you add Module + Factor

    Zero is never included in the data
    Will drop terms which are very small compared to the total magnitude
    """

    def __init__(
        self, basis_factor: dict[BaseBasis, Any], *, unity_basis, op_prio, normalize=None, clip_small=1e-10
    ) -> None:
        """
        op_prio: low is highest prio; less prio will pass operation to more prio
        """
        if clip_small is not None:
            max_factor = max(  # determine max. absolute value of factors in order to potential clip small values
                [
                    abs(factor)
                    for factor in basis_factor.values()
                    if isinstance(factor, numbers.Complex)
                    # hasattr(factor, "__abs__")
                ],
                default=None,
            )

        basis_factor = {
            basis: factor
            for basis, factor in basis_factor.items()
            if not is_zero(factor)
            and (
                clip_small is None
                or max_factor is None
                or not (
                    isinstance(factor, numbers.Complex)
                    # hasattr(factor, "__abs__")
                    and abs(factor) < max_factor * clip_small
                )
            )
        }

        self.basis_factor = normalize(basis_factor) if normalize else basis_factor
        self.normalize = normalize
        self.unity_basis = unity_basis
        self.op_prio = op_prio
        self.clip_small = clip_small

    def transform(self, func):  # currently not used much
        return self._create(func(self.basis_factor))

    def _create(self, basis_factor):
        """
        used to create results with appropriate initialization of the same properties
        """
        return self.__class__(
            basis_factor,
            unity_basis=self.unity_basis,
            op_prio=self.op_prio,
            normalize=self.normalize,
            clip_small=self.clip_small,
        )

    def __iter__(self):
        return iter(self.basis_factor.items())

    def _ensure_prio(self, other):
        """
        less prio Module are rejected
        more prio
        """
        do_take_operation = isinstance(other, numbers.Number) or (  # TODO: sometimes no sides wants the operation?
            hasattr(other, "op_prio") and other.op_prio >= self.op_prio
        )
        if not do_take_operation:
            return NotImplemented

        if not isinstance(other, self.__class__) or self.op_prio != other.op_prio:
            other = self._unity(other)

        return other

    def __add__(self, other) -> "Module":
        if other == 0:
            return self

        other_wrapped = self._ensure_prio(other)

        if other_wrapped is NotImplemented:
            if isinstance(other, self.__class__):  # because same class __radd__ would not be called due to Python
                return other.__radd__(self)

            return NotImplemented

        basis_factor = self.basis_factor.copy()

        for other_basis, other_factor in other_wrapped.basis_factor.items():
            if other_basis in basis_factor:
                basis_factor[other_basis] = basis_factor[other_basis] + other_factor
            else:
                basis_factor[other_basis] = other_factor

        return self._create(basis_factor)

    def __radd__(self, other) -> "Module":
        return self + other

    def __mul__(self, other: Factor) -> "Module":
        """
        support only multiplication with factor level
        strictly speaking this makes it a left+right module
        other factor always unchanged when going through the basis
        """
        if other == 1:
            return self

        basis_factor = {basis: factor * other for basis, factor in self.basis_factor.items()}

        return self._create(basis_factor)

    def __rmul__(self, other: Factor) -> "Module":
        if other == 1:
            return self

        basis_factor = {basis: other * factor for basis, factor in self.basis_factor.items()}

        return self._create(basis_factor)

    def __neg__(self):
        return self._create({basis: -factor for basis, factor in self.basis_factor.items()})

    def __eq__(self, other):
        if other == 0:
            return not self.basis_factor

        other = self._ensure_prio(other)

        if other is NotImplemented:
            return NotImplemented

        if isinstance(other, self.__class__):
            return self.basis_factor.keys() == other.basis_factor.keys() and all(
                self.basis_factor[basis] == other.basis_factor[basis] for basis in self.basis_factor.keys()
            )

        return self == self._unity(other)

    def __hash__(self):
        return hash(frozenset(self.basis_factor.items()))

    def __getitem__(self, basis):
        """
        for convenient deconstruction
        """
        return self.basis_factor[basis]

    @property
    def scalar_part(self):
        """
        return 0 if not explicit unity part
        """
        return sum(
            factor_scalar * basis_scalar
            for basis, factor in self.basis_factor.items()
            for factor_scalar, basis_scalar in [(getattr(factor, "scalar_part", factor), basis.scalar_part)]
            if basis_scalar != 0
        )

    def _unity(self, factor: Factor):  # TODO: annotate cls
        """
        Returns 1 element
        Needed update upgrade factors with a basis for operations
        e.g. when adding a scalar value
        """
        return self._create({self.unity_basis: factor})

    def _zero(self):  # rarely needed
        return self._create({})

    def _repr_pretty_(self, printer, cycle, support_newlines=True):
        if cycle:
            return printer.text("...")

        if not self.basis_factor:
            printer.text("0")
            return

        multi_line = support_newlines and len(self.basis_factor) > 1 and _is_long_repr(self.basis_factor)

        for i, (basis, factor) in enumerate(
            sorted(self.basis_factor.items(), key=lambda b_f: b_f[0]._sort_key())
        ):  # assumes basis has __lt__
            is_first_element = i == 0

            if is_negative(factor):
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

            do_print_factor = basis.is_unity() or not is_identity(factor)

            factor_needs_parenthesis = do_print_factor and _repr_needs_parenthesis(factor)

            if factor_needs_parenthesis:
                printer.begin_group(1, "(")

            if do_print_factor:
                printer.pretty(factor)

            if factor_needs_parenthesis:
                printer.end_group(1, ")")

            if not basis.is_unity():
                if do_print_factor:
                    printer.text(" ")

                printer.pretty(basis)

        if multi_line:
            printer.end_group(2)

    def __repr__(self):
        printer = ReprPrinter()
        self._repr_pretty_(printer, cycle=False, support_newlines=False)
        return "{ " + printer.value() + " }"

    @property
    def c(self):
        return self.conjugate()

    def linear_func(self, func_name):
        return self._create(
            {
                # fmt: off
                new_basis:
                new_factor
                # fmt: on
                for basis, factor in self.basis_factor.items()
                for new_basis, new_factor in [getattr(basis, func_name)(factor)]
                if new_factor != 0
            }
        )

    def conjugate(self):
        """
        takes conjugate separately on basis and factor (for performance)
        not appropriate when conjugate of basis introduces new factor
        """
        return self.linear_func("conjugate")


def _repr_needs_parenthesis(factor):
    # if isinstance(factor, numbers.Complex):
    #    return factor.real != 0 and factor.imag != 0

    if isinstance(factor, Module):
        return len(factor.basis_factor) > 1  # or (
        #    len(factor.basis_factor) == 1 and _repr_needs_parenthesis(next(iter(factor.basis_factor.values())))
        # )

    return False


def _is_long_repr(basis_factor):
    result = len(basis_factor) > MAX_ONE_LINE_ELEM or any(
        isinstance(factor, Module) and _is_long_repr(factor.basis_factor) for factor in basis_factor.values()
    )

    return result


class Algebra(Module):
    """
    adds multiplication between "Module elements"
    basis elements need to be able to multiply and yield Algebra object
    in multiplication only use Factor or Algebra; never a basis
    """

    def __mul__(self, other: "Algebra") -> "Algebra":
        """
        __rmul__ not needed since Algebra*Algebra
        or multiplied from the right?
        """
        other_wrapped = self._ensure_prio(other)

        if other_wrapped is NotImplemented:
            if isinstance(other, self.__class__):  # because same class __radd__ would not be called due to Python
                return other.__rmul__(self)
            return NotImplemented

        basis_factor = {}
        for (basis1, factor1), (basis2, factor2) in itertools.product(
            self.basis_factor.items(), other_wrapped.basis_factor.items()
        ):
            #####
            new_basis_factors = basis1.mul(factor1, basis2, factor2)
            #####

            for result_basis, result_factor in new_basis_factors.items():
                if result_basis not in basis_factor:
                    basis_factor[result_basis] = result_factor
                else:
                    basis_factor[result_basis] += result_factor

        result = self._create(basis_factor)

        return result

    def __rtruediv__(self, numer):
        """
        divide if only 1 element in sum
        and returns Quotient otherwise
        """
        if not self.basis_factor:
            raise ZeroDivisionError("Algebra is zero")

        if len(self.basis_factor) == 1:
            basis, factor = next(iter(self.basis_factor.items()))
            if hasattr(basis, "inverse") and (
                (inverse := basis.inverse()) is not NotImplemented
            ):  # in those cases there is a simplified inverse
                return numer * self._create(inverse) * (1 / factor)

        return Quotient(numer, self)
