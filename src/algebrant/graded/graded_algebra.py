import types
from typing import Self

from algebrant.algebra import Algebra
from algebrant.graded.graded_symbol import Graded


class GradedAlgebra(Algebra[Graded]):
    """
    basis elements must be CliffordBases with tuples
    """

    @property
    def i(self) -> Self:
        """
        Main involution

        satisfies (A * B).i = A.i * B.i
        """
        return self.flip_grade_signs(lambda x: x % 2 == 1)

    @property
    def r(self) -> Self:
        """
        Anti-involution
        Reversion - reverses multi-vectors (does not take conjugate)

        satisfies (A * B).r = B.r * A.r

        A*A.r cannot have grades 4k+{2,3}
        """
        return self.flip_grade_signs(lambda x: x % 4 in (2, 3))

    @property
    def cl(self) -> Self:
        """
        Anti-involution
        Clifford conjugate - combination of A.c = A.r.i

        satisfies (A * B).c = B.c * A.c

        A*A.c cannot have grades 4k+{1,2}
        """
        return self.flip_grade_signs(lambda x: x % 4 in (1, 2))

    def flip_grade_signs(self, sign_flip_condition) -> Self:
        basis_factor = {}
        for basis, factor in self.basis_factor.items():
            if sign_flip_condition(basis.grade):
                factor = -factor

            basis_factor[basis] = factor

        return self._new(basis_factor)

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
