import types
from typing import Self, TypeVar

from algebrant.algebra.algebra import Algebra
from algebrant.algebra.algebra_data import AlgebraData
from algebrant.graded.commute import commute
from algebrant.graded.graded_protocol import GradedProtocol

Factor = TypeVar("Factor")


class GradedAlgebra[Basis: GradedProtocol](Algebra):
    """
    basis elements must be CliffordBases with tuples
    """

    @property
    def i(self) -> Self:
        """
        Main involution

        satisfies (A * B).i = A.i * B.i
        """
        return self.map(
            lambda basis, factor: [
                (basis_i, getattr(factor, "i", factor) * factor_i) for basis_i, factor_i in basis.i
            ]
        )

    @property
    def r(self) -> Self:
        """
        Anti-involution
        Reversion - reverses multi-vectors (does not take conjugate)

        satisfies (A * B).r = B.r * A.r

        A*A.r cannot have grades 4k+{2,3}
        """
        return self.map(
            lambda basis, factor: [
                (basis_comm, factor_r * factor_comm)
                for basis_r, factor_r in basis.r
                for basis_comm, factor_comm in commute(basis_r, getattr(factor, "r", factor))
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
        return self.map(
            lambda basis, factor: [
                (basis_comm, factor_cl * factor_comm)
                for basis_cl, factor_cl in basis.cl
                for basis_comm, factor_comm in commute(basis_cl, getattr(factor, "cl", factor))
            ]
        )

    @property
    def scalar(self) -> Factor | float:
        """
        Returns the scalar part of the algebra, which is the part with grade 0.
        """
        result = [factor for basis, factor in self.basis_factor if basis.grade == 0]

        if len(result) > 1:
            raise ValueError(f"Expected only one scalar part, got {len(result)}: {result}")

        if len(result) == 0:
            return 0

        return result[0]

    def take_grades(self, *grades) -> Self:
        if grades and isinstance(grades[0], types.FunctionType):
            grade_test_func = grades[0]
        else:
            grade_test_func = lambda g: g in grades  # noqa: E731

        return self._new(  # TODO: generalize
            AlgebraData(
                {
                    basis: factor
                    for basis, factor in self.basis_factor
                    if grade_test_func(basis.grade)
                }
            )
        )

    @property
    def grades(self) -> frozenset[int | None]:  # TODO: remove somehow?
        return frozenset(basis.grade for basis, _ in self.basis_factor)

    def split_into_grades(self) -> dict[int | None, Self]:
        """
        Split the algebra into grades
        """
        return {grade: self.take_grades(grade) for grade in self.grades}

    @property
    def sqr(self):
        """
        returns square of element as scalar, but throws exception is square is not scalar
        """
        result = self * self
        if not result.grades <= {0}:
            raise ValueError(
                f".sqr is expected to be used only with elements which square to a scalar. however, the result had grades {self.grades}: {result}"
            )

        return result.scalar
