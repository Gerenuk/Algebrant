import itertools
import math
import types
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .algebra import Algebra
from .base_classes import BaseBasis
from .common import conjugate
from .repr_printer import ReprPrinter

try:
    import colorful

    colorful.use_true_colors()
    vec_col = colorful.deepSkyBlue
except ImportError:
    vec_col = lambda x: x


"""
Todos (some old?):
* implement conjugate for blades squaring to -1 (needed for .c)
* expand in non-orthogonal basis too
* single element sums unpacked? esp before division
"""


def sqr_to_scalar(val: "CliffordAlgebra"):
    """
    Returns (scalar**2, non_scalar**2) but non_scalar only if it squares to a scalar
    """
    non_scalar = val.take_grades(lambda g: g > 0)

    bases, values = zip(*non_scalar)
    all_anti_commute = not any(b1.commutes_with(b2) for b1, b2 in itertools.combinations(bases, r=2))

    non_scalar_sqr = sum(value**2 * base.sqr() for base, value in zip(bases, values)) if all_anti_commute else None

    return val.scalar_part**2, non_scalar_sqr


@dataclass(unsafe_hash=True)
class CliffordBasis(BaseBasis):
    bases: tuple  # needs to be sorted (or will be sorted by force)

    def __post_init__(self):
        if list(self.bases) != sorted(self.bases):
            raise ValueError("Bases must be sorted")

    @classmethod
    def unity(cls):
        return cls(tuple())

    @property
    def scalar_part(self):
        if self.bases == tuple():
            return 1

        return 0

    def is_unity(self):
        return self.bases == tuple()

    def _sort_key(self):
        return (len(self.bases), self.bases)

    def __mul__(self, other: "CliffordBasis") -> dict:
        common_bases = set(self.bases) & set(other.bases)

        if any(self._vector_squares_to_zero(basis) for basis in common_bases):
            return {}

        result_bases, is_negative = ga_basis_mul(self.bases, other.bases)

        num_squares_to_minus_1 = sum(self._vector_squares_to_minus_1(basis) for basis in common_bases)
        if num_squares_to_minus_1 % 2 == 1:
            is_negative = not is_negative

        return {CliffordBasis(tuple(result_bases)): 1 if not is_negative else -1}

    def __xor__(self, other: "CliffordBasis") -> dict:
        """
        Wedge product
        """
        if set(self.bases) & set(other.bases):  # same as normal product unless some vectors in common
            return {}

        return self * other

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            return printer.text("...")

        if not self.bases:
            printer.text("(1)")  # should usually not be shown
            return

        for i, basis in enumerate(self.bases):
            if i > 0:
                printer.text(" ")

            printer.pretty(vec_col(basis))  # TODO: later Basis objects?

    def __repr__(self):
        printer = ReprPrinter()
        self._repr_pretty_(printer, cycle=False)
        return printer.value()

    def commutes_with(self, other: "CliffordBasis"):
        len_self = len(self.bases)
        len_other = len(other.bases)
        len_common = len(set(self.bases) & set(other.bases))

        return (len_self * len_other + len_common) % 2 == 0

    @property
    def grade(self):
        return len(self.bases)

    def sqr(self):
        is_negative = len(self.bases) % 4 in (2, 3)

        # check for vectors squaring to -1
        num_negative_squares = sum(self._vector_squares_to_minus_1(basis) for basis in self.bases)
        if num_negative_squares % 2 == 1:  # assuming negative basis vector change sign
            is_negative = not is_negative

        return -1 if is_negative else 1

    def conjugate(self, factor):
        if len(self.bases) % 4 in (2, 3):
            factor_sign = -1
        else:
            factor_sign = 1

        # check for vectors squaring to -1
        num_negative_squares = sum(self._vector_squares_to_minus_1(basis) for basis in self.bases)
        if num_negative_squares % 2 == 1:  # assuming negative basis vector change sign
            factor_sign = -factor_sign

        return self, factor_sign * factor.conjugate()

    @staticmethod
    def _vector_squares_to_minus_1(basis_elem):
        """
        For a single vector element, not the whole basis blade

        convention that capital first letter squares to -1
        """
        return isinstance(basis_elem, str) and len(basis_elem) >= 1 and basis_elem[0].isupper()

    @staticmethod
    def _vector_squares_to_zero(basis_elem):
        return isinstance(basis_elem, str) and len(basis_elem) >= 1 and basis_elem[0] == "_"


class CliffordAlgebra(Algebra):
    """
    basis elements must be CliffordBases with tuples
    """

    @property
    def i(self):
        """
        Main involution

        satisfies (A * B).i = A.i * B.i
        """
        return self.flip_grade_signs(lambda x: x % 2 == 1)

    @property
    def r(self):
        """
        Anti-involution
        Reversion - reverses multi-vectors (does not take conjugate)

        satisfies (A * B).r = B.r * A.r

        A*A.r cannot have grades 4k+{2,3}
        """
        return self.flip_grade_signs(lambda x: x % 4 in (2, 3))

    @property
    def cl(self):
        """
        Anti-involution
        Clifford conjugate - combination of A.c = A.r.i

        satisfies (A * B).c = B.c * A.c

        A*A.c cannot have grades 4k+{1,2}
        """
        return self.flip_grade_signs(lambda x: x % 4 in (1, 2))

    def flip_grade_signs(self, sign_flip_condition, factor_conjugate=False):
        basis_factor = {}
        for basis, factor in self.basis_factor.items():
            if factor_conjugate:
                factor = conjugate(factor)
                basis, extra_factor = conjugate(basis)  # e.g. for blades squaring to -1
                factor *= extra_factor  # TODO: like this?

            if sign_flip_condition(basis.grade):
                factor = -factor

            basis_factor[basis] = factor

        return self._create(basis_factor)

    def __rtruediv__(self, numer):
        """
        guaranteed to work only up to dimension 5 or special grade cases
        """

        # TODO: rule for algebra split
        # e.g. split into A(1+I)+B(1-I) or A(1+iI)+B(1-iI) and I is product of all bases
        # where A,B have single (anti)commutation type with I
        # -> you can remove some basis vectors and split algebra
        # -> works when I odd or numer fully even or numer fully odd

        # TODO: flip signs if result would be "simpler"
        # e.g. (1+e1234+e3456) (1+e1234-e3456)=1 + 2 e1234

        # TODO:
        # special rule for when there is no scalar (extract lowest grade having frequent basis?)

        # TODO: Advanced sign-flip:
        # split blades into groups A, B such that within group A/B all anti-commute and across groups A to B all commute
        # -> flip sign of 1 group
        # (!) but no gain if scalar is present

        # TODO: speed up A*A.r case by exploiting the cancellation structure

        # dim>=6 you cannot generally solves grades {0,3,4}

        if not self.basis_factor:
            raise ZeroDivisionError("Division by Multivector zero")

        grades = self.grades

        #################################### Divide by plain scalar
        if grades == {0}:
            # print("Scalar")
            return numer * (1 / self.scalar_part)

        #################################### Test if flipping non-scalar is enough
        scalar_sqr, non_scalar_sqr = sqr_to_scalar(self)

        if non_scalar_sqr is not None:
            # print("Anticommuting")
            scalar = scalar_sqr - non_scalar_sqr
            # is this correct for vectors squaring to -1?

            if scalar == 0:
                raise ZeroDivisionError(f"Zero division 1/((s+M)(s-M)) for {self}")

            return numer * self.flip_grade_signs(lambda g: g > 0) * (1 / scalar)  # division on scalars here

        #################################### Special rule for grades {0,1,dim-1,dim}
        # this can save 1 step instead of doing Clifford conjugation and also is needed for higher even dim.

        dimension = len(set(itertools.chain.from_iterable(basis.bases for basis in self.basis_factor.keys())))

        if grades <= {0, 1, dimension - 1, dimension}:
            # print("Special {0,1,dim-1,dim}")

            if dimension % 2 == 1:
                flip_grades = (1, dimension - 1)
            else:
                flip_grades = (1,)

            flipped_sign_mv = self.flip_grade_signs(lambda g: g in flip_grades)

            new_inverse = self * flipped_sign_mv

            if new_inverse == 0:
                raise ZeroDivisionError(f"Zero division 1/(A*i(A; {','.join(map(str, flip_grades))})) for {self}")

            if new_inverse.grades == {0}:
                inv_of_new_inverse = 1 / new_inverse.scalar_part
            else:
                # print("Anticommuting")
                # do the non-scalar-flip inversion right away as it's the next step
                scalar_sqr, non_scalar_sqr = sqr_to_scalar(new_inverse)

                assert non_scalar_sqr is not None, non_scalar_sqr

                scalar = scalar_sqr - non_scalar_sqr

                if scalar == 0:
                    raise ZeroDivisionError(f"Zero division 1/((s+M)(s-M)) for {new_inverse}")

                inv_of_new_inverse = new_inverse.flip_grade_signs(lambda g: g > 0) * (1 / scalar)

            return numer * flipped_sign_mv * inv_of_new_inverse

        #################################### test if grades 4k+{1,2} which I could reduce; only for dimension >= 6
        grades_mod_4 = Counter(g % 4 for g in self.grades)

        if dimension >= 6 and grades_mod_4[1] > grades_mod_4[3]:
            # print("Clifford reverse")
            new_inverse = self * self.c

            assert not any(
                g % 4 in (1, 2) for g in new_inverse.grades
            ), f"Grades 1,2 did not reduce with result {new_inverse} (should not happen unless due to floating point uncertainty)"

            if new_inverse == 0:
                raise ZeroDivisionError(f"Zero division 1/(A*A.c) for {self}")

            if new_inverse.grades == {0}:
                new_inverse = new_inverse.scalar_part

            if all(g % 4 != 3 for g in new_inverse.grades):  # otherwise may be infinite loop with 4k+{2,3} rule
                return numer * self.c * (1 / new_inverse)

        #################################### test if grades 4k+{2,3} which I could reduce
        if grades_mod_4.keys() & {2, 3}:
            # print("Reverse")
            new_inverse = self * self.r

            assert not any(
                g % 4 in (2, 3) for g in new_inverse.grades
            ), f"Grades 2,3 did not reduce with result {new_inverse} (should not happen unless due to floating point uncertainty)"

            if new_inverse == 0:
                raise ZeroDivisionError(f"Zero division 1/(A*A.r) for {self}")

            if new_inverse.grades == {0}:
                new_inverse = new_inverse.scalar_part

            return numer * self.r * (1 / new_inverse)

        raise NotImplementedError(f"Cannot divide by multivector with grades {self.grades}: {self} ")

    def __rmatmul__(self, other):
        result = (conjugate(other) * self).scalar_part

        return result

    def __matmul__(self, other):
        """
        sesquilinear product (i.e. not symmetric)
        swapping arguments takes complex conjugate
        """
        result = (self.c * other).scalar_part

        return result

    def __abs__(self):
        abs_sqr = sum(scalar * conjugate(scalar) for _base, scalar in self)

        if isinstance(abs_sqr, (complex, np.complex_)):
            assert abs_sqr == 0 or abs_sqr.imag < abs(abs_sqr) * 1e-10, abs_sqr  # TODO: not hard-code?

            abs_sqr = abs_sqr.real  # since should always be real

        return math.sqrt(abs_sqr)

    def __xor__(self, other):
        """
        assumes other is cliffordalgebra
        """
        other = self._ensure_prio(other)

        if other is NotImplemented:
            return NotImplemented

        basis_factor = {}
        for (basis1, factor1), (basis2, factor2) in itertools.product(
            self.basis_factor.items(), other.basis_factor.items()
        ):
            new_elem = factor1 * factor2 * self._create(basis1 ^ basis2)  # only change compared to __mul__
            for result_basis, result_factor in new_elem.basis_factor.items():
                if result_basis not in basis_factor:
                    basis_factor[result_basis] = result_factor
                else:
                    basis_factor[result_basis] += result_factor

        return self._create(basis_factor)

    def take_grades(self, *grades):
        if grades and isinstance(grades[0], types.FunctionType):
            grade_test_func = grades[0]
        else:
            grade_test_func = lambda g: g in grades

        return self._create(
            {basis: factor for basis, factor in self.basis_factor.items() if grade_test_func(basis.grade)}
        )

    @property
    def grades(self):
        return frozenset(basis.grade for basis in self.basis_factor.keys())


def ga_basis_mul(basis1, basis2):
    """
    Clifford algebra product of basis vectors
    basis1, basis2: sorted list of elements of grade-1 vectors; e.g. "e124" = (1,2,4) or ("e1", "e2", "e4")
    :return: (tuple of basis indices, is_negative); scalar

    Does not do negative signature
    basis1 and basis2 are essential merged to a sorted results without duplicates
    """
    assert list(basis1) == sorted(basis1), basis1  # TODO: remove?
    assert list(basis2) == sorted(basis2), basis2

    basis = []
    is_negative = 0

    i1 = 0
    i2 = 0

    n1 = len(basis1)
    n2 = len(basis2)

    xor2 = 0  # toggled by b; will itself toggle the sign by a

    while i1 < n1 and i2 < n2:
        cur1 = basis1[i1]
        cur2 = basis2[i2]

        if cur1 <= cur2:
            if cur1 != cur2:
                basis.append(cur1)
            i1 += 1
            is_negative ^= xor2

        if cur2 <= cur1:
            if cur1 != cur2:
                basis.append(cur2)
            i2 += 1
            xor2 ^= 1

    is_negative ^= (n1 - i1) & xor2 & 1  # remaining sign changes

    if i1 < n1:
        basis.extend(basis1[i1:])
    elif i2 < n2:  # cannot be both at the same time
        basis.extend(basis2[i2:])

    return basis, is_negative


def mv_sqrt(A):
    """
    TODO: sometimes numerical issues; better reverse
    """
    from cmath import sqrt

    ns = A.take_grades(lambda g: g > 0)

    scalar_sqr, non_scalar_sqr = sqr_to_scalar(A)

    if non_scalar_sqr is None:
        raise NotImplementedError(f"Cannot take sqrt of {A}")

    scalar = A.scalar_part

    if non_scalar_sqr == 0:
        c1 = sqrt(scalar)
        c2 = -c1
    else:
        b = sqrt(scalar_sqr - non_scalar_sqr)
        c1 = sqrt((scalar + b) / 2)
        c2 = sqrt((scalar - b) / 2)

    return (
        c1 + ns / (2 * c1),
        c2 + ns / (2 * c2),
    )
