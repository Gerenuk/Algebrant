import functools
import itertools
from typing import Self, Sequence

from algebrant.clifford.clifford_algebra import CliffordAlgebra
from algebrant.clifford.clifford_basis import CliffordBasis
from algebrant.clifford.clifford_basis_vec import CliffordBasisVec
from algebrant.random_mv import random_from_basis


class ClAlg:
    def __init__(
        self,
        bases: tuple[CliffordBasisVec, ...],
        complex: bool = True,
        name: str | None = None,
    ) -> None:
        self.bases = bases
        self.is_complex: bool = complex
        self.name: str = (
            name
            if name is not None
            else "Cl("
            + ",".join(map(str, bases))
            + ","
            + ("C" if complex else "R")
            + ")"
        )

    @classmethod
    def from_pq(cls, p, q=0, start_idx: int = 1) -> Self:
        bases = [CliffordBasisVec(f"E{start_idx + i}", sqr=-1) for i in range(q)] + [
            CliffordBasisVec(f"e{start_idx + q + i}", sqr=1) for i in range(p)
        ]
        if q != 0:
            name: str = f"Cl({p},{q},R)"
        else:
            name = f"Cl({p},R)"
        return cls(tuple(bases), complex=False, name=name)

    @classmethod
    def from_dim(cls, dim, *, start_idx: int = 1) -> Self:
        bases = [CliffordBasisVec(f"e{start_idx + i}", sqr=1) for i in range(dim)]
        return cls(tuple(bases), complex=True, name=f"Cl({dim},C)")

    def get_bases(self, *grades) -> Sequence[CliffordAlgebra]:
        if not grades:
            grades = range(len(self.bases) + 1)

        result = []

        if 0 in grades:
            result.append(CliffordAlgebra({}, basis_class=CliffordBasis))

        result.extend(
            CliffordAlgebra({CliffordBasis(basis): 1}, basis_class=CliffordBasis)
            for grade in grades
            for basis in itertools.combinations(self.bases, r=grade)
            if grade != 0
        )

        return result

    def rand(self, *grades, int_only=False, min_int=2, max_int=1000) -> CliffordAlgebra:
        bases = self.get_bases(*grades)

        return random_from_basis(
            bases,
            complex=self.is_complex,
            int_only=int_only,
            min_int=min_int,
            max_int=max_int,
        )

    def make_perp(self, *vecs) -> CliffordAlgebra:
        v = self.rand(len(vecs) + 1)
        result = functools.reduce(lambda x, y: (y << x), vecs, v)
        return result

    @property
    def I(self) -> CliffordAlgebra:
        return CliffordAlgebra(
            {CliffordBasis(self.bases): 1}, basis_class=CliffordBasis
        )

    @property
    def one(self) -> CliffordAlgebra:
        return CliffordAlgebra({CliffordBasis(tuple()): 1}, basis_class=CliffordBasis)

    def __repr__(self) -> str:
        return self.name
