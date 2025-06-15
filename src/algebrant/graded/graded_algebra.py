import types
from typing import Self, TypeVar

from algebrant.algebra.algebra import Algebra

GradedBasis = TypeVar("GradedBasis")


class GradedAlgebra[GradedBasis](Algebra):
    """
    basis elements must be CliffordBases with tuples
    """

    @staticmethod
    def _basis_factor_conjugate(basis_factor):
        basis, factor = basis_factor

        if hasattr(factor, "vector_conjugate") and basis.is_odd:
            return (basis, factor.vector_conjugate)

        return (basis, factor)

    @property
    def i(self) -> Self:
        """
        Main involution

        satisfies (A * B).i = A.i * B.i
        """
        return self.map_basisfactor(
            lambda bf: [
                (basis, bf[1].i * extra_factor) for basis, extra_factor in [bf[0].i]
            ]
        )
        # return self.flip_grade_signs(lambda x: x % 2 == 1)

    @property
    def r(self) -> Self:
        """
        Anti-involution
        Reversion - reverses multi-vectors (does not take conjugate)

        satisfies (A * B).r = B.r * A.r

        A*A.r cannot have grades 4k+{2,3}
        """
        return self.map_basisfactor(
            lambda bf: [
                self._basis_factor_conjugate((basis, bf[1].r * extra_factor))
                for basis, extra_factor in [bf[0].r]
            ]
        )

    @property
    def cl(self) -> Self:
        """
        Anti-involution
        Clifford conjugate - combination of A.c = A.r.i

        satisfies (A * B).c = B.c * A.c

        A*A.c cannot have grades 4k+{1,2}
        """
        return self.map_basisfactor(
            lambda bf: [
                self._basis_factor_conjugate((basis, bf[1].cl * extra_factor))
                for basis, extra_factor in [bf[0].cl]
            ]
        )

    def take_grades(self, *grades) -> Self:
        if grades and isinstance(grades[0], types.FunctionType):
            grade_test_func = grades[0]
        else:
            grade_test_func = lambda g: g in grades  # noqa: E731

        return self._new(
            {
                basis: factor
                for basis, factor in self.basis_factor.items()
                if grade_test_func(basis.grade)
            }
        )

    @property
    def grades(self) -> frozenset[int | None]:
        return frozenset(basis.grade for basis in self.basis_factor.keys())

    def split_into_grades(self) -> dict[int | None, Self]:
        """
        Split the algebra into grades
        """
        return {grade: self.take_grades(grade) for grade in self.grades}
