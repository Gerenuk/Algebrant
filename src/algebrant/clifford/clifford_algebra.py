import itertools
import math
from collections import Counter
from collections.abc import Iterable
from types import NotImplementedType
from typing import Any, Protocol, Self, Sequence, TypeVar

from algebrant.algebra.algebra_utils import MultiplicationMixin
from algebrant.clifford.clifford_basis import CliffordBasis
from algebrant.clifford.clifford_basis_vec import CliffordBasisVec
from algebrant.graded.graded_algebra import GradedAlgebra

BasisFactor = tuple[CliffordBasis, Any]


class CliffordAlgebra(GradedAlgebra[CliffordBasis], MultiplicationMixin):
    def _new(self, basis_factor: dict[CliffordBasis, Any]) -> Self:
        """
        used to create results with appropriate initialization of the same properties
        """
        return self.__class__(  # TODO: generalize?
            basis_factor={
                basis: factor
                for basis, factor in basis_factor.items()
                if abs(factor) > 1e-10  # TODO
            },
            basis_class=self.basis_class,
            op_prio=self.op_prio,
        )

    def _mul(
        self, bf1: BasisFactor, bf2: BasisFactor
    ) -> Iterable[tuple[BasisFactor, Any]]:
        basis1, factor1 = bf1
        basis2, factor2 = bf2

        result_bases, is_negative = ga_basis_mul(basis1.bases, basis2.bases)
        sign = [1, -1][is_negative]

        common_bases = set(basis1.bases) & set(basis2.bases)
        sign *= math.prod(b.sqr for b in common_bases)

        return [(self.basis_class(tuple(result_bases)), sign * factor1 * factor2)]

    def __mul__(self, other: Any) -> Self | NotImplementedType:
        return self._multiply(other, self._mul)

    def _xor(
        self, bf1: BasisFactor, bf2: BasisFactor
    ) -> Iterable[tuple[BasisFactor, Any]]:
        if set(bf1[0].bases) & set(bf2[0].bases):
            return []

        return self._mul(bf1, bf2)

    def __xor__(self, other: Any) -> Self | NotImplementedType:
        return self._multiply(other, self._xor)

    def _lshift(
        self, bf1: BasisFactor, bf2: BasisFactor
    ) -> Iterable[tuple[BasisFactor, Any]]:
        if not set(bf1[0].bases) <= set(bf2[0].bases):
            return []

        return self._mul(bf1, bf2)

    def __lshift__(self, other: Any) -> Self | NotImplementedType:
        return self._multiply(other, self._lshift)

    def _rshift(
        self, bf1: BasisFactor, bf2: BasisFactor
    ) -> Iterable[tuple[BasisFactor, Any]]:
        if not set(bf1[0].bases) >= set(bf2[0].bases):
            return []

        return self._mul(bf1, bf2)

    def __rshift__(self, other: Any) -> Self | NotImplementedType:
        return self._multiply(other, self._rshift)

    @property
    def scalar(self) -> Any:
        return self.basis_factor.get(CliffordBasis(tuple()), 0)

    def __rtruediv__(self, numer):
        # TODO: rule for algebra split
        # e.g. split into A(1+I)+B(1-I) or A(1+iI)+B(1-iI) and I is product of all bases
        # where A,B have single (anti)commutation type with I
        # -> you can remove some basis vectors and split algebra
        # -> works when I odd or numer fully even or numer fully odd

        # TODO: flip signs if result would be "simpler"
        # e.g. (1+e1234+e3456) (1+e1234-e3456)=1 + 2 e1234

        # TODO:
        # special rule for when there is no scalar (extract lowest grade having frequent basis?)

        # TODO: speed up A*A.r case by exploiting the cancellation structure

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

            return (
                numer * self.flip_grade_signs(lambda g: g > 0) * (1 / scalar)
            )  # division on scalars here

        #################################### Special rule for grades {0,1,dim-1,dim}
        # this can save 1 step instead of doing Clifford conjugation and also is needed for higher even dim.

        dimension = len(
            set(
                itertools.chain.from_iterable(
                    basis.bases for basis in self.basis_factor.keys()
                )
            )
        )

        if grades <= {0, 1, dimension - 1, dimension}:
            # print("Special {0,1,dim-1,dim}")

            if dimension % 2 == 1:
                flip_grades = (1, dimension - 1)
            else:
                flip_grades = (1,)

            flipped_sign_mv = self.flip_grade_signs(lambda g: g in flip_grades)

            new_inverse = self * flipped_sign_mv

            if new_inverse == 0:
                raise ZeroDivisionError(
                    f"Zero division 1/(A*i(A; {','.join(map(str, flip_grades))})) for {self}"
                )

            if new_inverse.grades == {0}:
                inv_of_new_inverse = 1 / new_inverse.scalar_part
            else:
                # print("Anticommuting")
                # do the non-scalar-flip inversion right away as it's the next step
                scalar_sqr, non_scalar_sqr = sqr_to_scalar(new_inverse)

                assert non_scalar_sqr is not None, non_scalar_sqr

                scalar = scalar_sqr - non_scalar_sqr

                if scalar == 0:
                    raise ZeroDivisionError(
                        f"Zero division 1/((s+M)(s-M)) for {new_inverse}"
                    )

                inv_of_new_inverse = new_inverse.flip_grade_signs(lambda g: g > 0) * (
                    1 / scalar
                )

            return numer * flipped_sign_mv * inv_of_new_inverse

        #################################### test if grades 4k+{1,2} which I could reduce; only for dimension >= 6
        grades_mod_4 = Counter(g % 4 for g in self.grades)

        # if dimension >= 6 and grades_mod_4[1] > grades_mod_4[3]:
        #     # print("Clifford reverse")
        #     new_inverse = self * self.c

        #     assert not any(
        #         g % 4 in (1, 2) for g in new_inverse.grades
        #     ), f"Grades 1,2 did not reduce with result {new_inverse} (should not happen unless due to floating point uncertainty)"

        #     if new_inverse == 0:
        #         raise ZeroDivisionError(f"Zero division 1/(A*A.c) for {self}")

        #     if new_inverse.grades == {0}:
        #         new_inverse = new_inverse.scalar_part

        #     if all(g % 4 != 3 for g in new_inverse.grades):  # otherwise may be infinite loop with 4k+{2,3} rule
        #         return numer * self.c * (1 / new_inverse)

        #################################### test if grades 4k+{2,3} which I could reduce
        if dimension <= 5 and grades_mod_4.keys() & {2, 3}:
            # print("Reverse")
            new_inverse = self * self.r

            assert not any(g % 4 in (2, 3) for g in new_inverse.grades), (
                f"Grades 2,3 did not reduce with result {new_inverse} (should not happen unless due to floating point uncertainty)"
            )

            if new_inverse == 0:
                raise ZeroDivisionError(f"Zero division 1/(A*A.r) for {self}")

            if new_inverse.grades == {0}:
                new_inverse = new_inverse.scalar_part

            return numer * self.r * (1 / new_inverse)

        ################################# the following works generally
        # TODO: replace since numerical issues

        max_steps = 2 ** math.floor((dimension + 1) / 2)
        inv = self
        for i in range(1, max_steps + 1):
            A = inv - max_steps * inv.scalar_part / i
            inv = self * A  # will clip small coefficients
            if inv.grades <= {0}:
                break
        else:
            raise NotImplementedError(
                f"Iterations of general inversion algorithm failed with {i=} {inv=} {A=}"
            )

        if inv == 0:
            raise ZeroDivisionError("Division by zero in general algorithm")

        return A / inv.scalar_part


class Sortable(Protocol):
    def __lt__(self, other: Self) -> bool: ...

    def __le__(self, other: Self) -> bool: ...


BasisElem = TypeVar("BasisElem", bound=Sortable)


def ga_basis_mul(
    basis1: Sequence[BasisElem], basis2: Sequence[BasisElem]
) -> tuple[Sequence[BasisElem], int]:
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


def Cl_vec(*names: str | int) -> CliffordAlgebra:
    str_names = [name if isinstance(name, str) else f"e{name}" for name in names]

    return CliffordAlgebra(
        {
            CliffordBasis(
                tuple(
                    CliffordBasisVec(name=name, sqr=1 if not name[0].isupper() else -1)
                    for name in str_names
                )
            ): 1
        },
        basis_class=CliffordBasis,
    )


def Cl_basis_vec(name: str | int, sqr=1) -> CliffordAlgebra:
    str_name = name if isinstance(name, str) else f"e{name}"

    return CliffordAlgebra(
        {CliffordBasis((CliffordBasisVec(name=str_name, sqr=sqr),)): 1},
        basis_class=CliffordBasis,
    )
