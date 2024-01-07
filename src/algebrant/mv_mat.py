from functools import reduce
from itertools import combinations

import numpy as np

from .creation import E
from .vector_basis import VecBasis

Paulis = [
    np.array([[1, 0], [0, 1]]),
    np.array([[1, 0], [0, -1]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, 1j], [-1j, 0]]),
]


def make_mat_basis_from_pauli(indices):
    return [reduce(np.kron, [Paulis[i] for i in e_idx]) for e_idx in indices]


def make_mat_basis(n):
    """
    for 2n dimensional Clifford algebra
    will create 2n+1 anti-commuting matrices (incl. one for the pseudoscalar)

    only first [:n] matrices really needed, but you may choose the last one instead too
    """
    indices = []
    for i in reversed(range(n)):
        for a in [1, 2]:
            idx = (0,) * i + (a,) + (3,) * (n - 1 - i)
            indices.append(idx)

    indices.append((3,) * n)  # extra matrix to choose from

    return make_mat_basis_from_pauli(indices)


class CliffordMatrixRepr:
    def __init__(self, vec_mats):
        clifford_dim = len(vec_mats)

        self.mv_basis = VecBasis(
            [E(*idxs) for k in range(0, clifford_dim + 1) for idxs in combinations(range(1, clifford_dim + 1), r=k)],
            dot=lambda a, b: (a.c * b).scalar_part,
        )

        mat_dim = vec_mats[0].shape[0]

        self.mat_basis = VecBasis(
            [np.matrix(np.identity(mat_dim))]
            + [
                reduce(np.dot, [vec_mats[i] for i in idxs])
                for k in range(1, clifford_dim + 1)
                for idxs in combinations(range(len(vec_mats)), r=k)
            ],
            dot=lambda a, b: np.trace(a.conj().T @ b),
        )

    @classmethod
    def from_dim(cls, dim):
        if not dim % 2 == 0:
            raise ValueError("Only even dimensions supported (odd dimensions require a direct product of 2 matrices)")

        n = dim // 2
        vec_mats = make_mat_basis(n)[:dim]
        return cls(vec_mats)

    def to_mv(self, val):
        return self.mv_basis.to_elem(self.mat_basis.to_vec(val))

    def to_mat(self, val):
        return self.mat_basis.to_elem(self.mv_basis.to_vec(val))


def make_cl4_chiral():
    Cl4_chiral_pauli_idx = [(3, 0), (2, 1), (2, 2), (2, 3)]  # fully off-diagonal

    vec_mats = make_mat_basis_from_pauli(Cl4_chiral_pauli_idx)

    return CliffordMatrixRepr(vec_mats)
