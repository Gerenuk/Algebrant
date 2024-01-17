import itertools
import numbers
from math import prod
from operator import itemgetter
from typing import Any

from .common import conjugate, is_identity, is_zero
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


def is_negative(val):
    """
    used to determine whether to translate "... + -a" into "... - a"
    """
    return (
        (isinstance(val, numbers.Real) and val < 0)
        or (isinstance(val, numbers.Complex) and (val.real < 0 or (val.real == 0 and val.imag < 0)))
        or (
            isinstance(val, Algebra)
            and val.basis_factor
            and all(is_negative(factor) for factor in val.basis_factor.values())
        )
    )


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

    def __init__(self, basis_factor: dict, *, unity_basis, op_prio, normalize=None, clip_small=1e-10) -> None:
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

    def transform(self, func):
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

    def __add__(self, other: "Module" | Factor) -> "Module":
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

    def __radd__(self, other: "Module" | Factor) -> "Module":
        return self + other

    def __mul__(self, other: Factor) -> "Module":
        """
        support only multiplication with factor level
        """
        basis_factor = {basis: factor * other for basis, factor in self.basis_factor.items()}

        return self._create(basis_factor)

    def __rmul__(self, other: Factor) -> "Module":
        basis_factor = {basis: other * factor for basis, factor in self.basis_factor.items()}

        return self._create(basis_factor)

    def __neg__(self):
        return self._create({basis: -factor for basis, factor in self.basis_factor.items()})

    def __eq__(self, other: "Module" | Factor):
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

    def _unity(self, factor: Factor):  # TODO: annotate cls
        """
        Returns 1 element
        Needed update upgrade factors with a basis for operations
        """
        return self._create({self.unity_basis: factor})

    def _zero(self):  # rarely needed
        return self._create({})

    def _is_long_repr(self):
        return len(self.basis_factor) > MAX_ONE_LINE_ELEM or any(
            getattr(factor, "_is_long_repr", False) for factor in self.basis_factor.values()
        )

    def _repr_pretty_(self, printer, cycle, support_newlines=True):
        if cycle:
            return printer.text("...")

        if not self.basis_factor:
            printer.text("0")
            return

        long_algebra = support_newlines and self._is_long_repr()

        for i, (basis, factor) in enumerate(
            sorted(self.basis_factor.items(), key=itemgetter(0))
        ):  # assumes basis has __lt__
            is_first_element = i == 0

            if is_negative(factor):
                if long_algebra:
                    if is_first_element:
                        printer.text(" -")
                    else:
                        printer.text("- ")
                else:
                    if is_first_element:
                        printer.text("-")
                    else:
                        printer.text(" - ")

                factor = -factor
            else:
                if long_algebra:
                    if is_first_element:
                        printer.text("  ")
                    else:
                        printer.text("+ ")
                else:
                    if not is_first_element:
                        printer.text(" + ")

            if basis == self.unity_basis:
                printer.pretty(factor)
            elif is_identity(factor):
                printer.pretty(basis)
            else:
                factor_needs_parenthesis = (isinstance(factor, Module) and len(factor.basis_factor) > 1) or (
                    isinstance(factor, numbers.Complex) and factor.real != 0 and factor.imag != 0
                )

                if factor_needs_parenthesis:
                    # printer.text("(")
                    printer.begin_group(3, "(")
                printer.pretty(factor)
                if factor_needs_parenthesis:
                    # printer.text(")")
                    printer.end_group(3, ")")

                if not basis == self.unity_basis:
                    printer.text(" ")
                    printer.pretty(basis)

            if long_algebra:
                printer.break_()

        # if long_algebra:
        #    printer.end_group(3)

    def __repr__(self):
        printer = ReprPrinter()
        self._repr_pretty_(printer, cycle=False, support_newlines=False)
        # return "{" + printer.value() + f";p{self.op_prio}}}"
        return "{" + printer.value() + "}"

    def __invert__(self):
        return self.conjugate()

    @property
    def c(self):
        return self.conjugate()

    def conjugate(self):
        """
        takes conjugate separately on basis and factor (for performance)
        not appropriate when conjugate of basis introduces new factor
        """
        return self._create(
            {
                # fmt: off
                conjugate(basis):
                conjugate(factor)
                # fmt: on
                for basis, factor in self.basis_factor.items()
            }
        )


class Algebra(Module):
    """
    adds multiplication between "Module elements"
    basis elements need to be able to multiply and yield Algebra object
    in multiplication only use Factor or Algebra; never a basis
    """

    def __mul__(self, other: "Algebra") -> "Algebra":
        """
        __rmul__ not needed since Algebra*Algebra
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
            new_elem = factor1 * factor2 * self._create(basis1 * basis2)

            for result_basis, result_factor in new_elem.basis_factor.items():
                if result_basis not in basis_factor:
                    basis_factor[result_basis] = result_factor
                else:
                    basis_factor[result_basis] += result_factor

        return self._create(basis_factor)

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
