from functools import reduce
from itertools import combinations

import numpy as np

from .common import conjugate, scalar_part
from .creation import E
from .vector_basis import ConvertVecBasis, VecBasis

Paulis = [
    np.array([[1, 0], [0, 1]]),  # identity
    np.array([[1, 0], [0, -1]]),  # sigma_z
    np.array([[0, 1], [1, 0]]),  # sigma_x
    np.array([[0, 1j], [-1j, 0]]),  # -sigma_y
]


def make_mat_basis_from_pauli(indices):
    return [reduce(np.kron, [Paulis[i] for i in e_idx]) for e_idx in indices]


def make_anti_comm_mat(n: int):
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


def make_cl_mat_basis(dim: int):
    if not dim % 2 == 0:
        raise ValueError("Only even dimensions supported (odd dimensions require a direct product of 2 matrices)")

    n = dim // 2
    vec_mats = make_anti_comm_mat(n)[:dim]
    return vec_mats


def make_mv_vec_basis(dim, start_idx=1):
    return VecBasis(
        [E(*idxs) for k in range(0, dim + 1) for idxs in combinations(range(start_idx, start_idx + dim), r=k)],
        dot=lambda a, b: scalar_part(conjugate(a) * b),
    )


def make_cl_vec_basis(vec_mats):
    clifford_dim = len(vec_mats)

    mat_dim = vec_mats[0].shape[0]

    mat_basis = VecBasis(
        [np.matrix(np.identity(mat_dim))]
        + [
            reduce(np.dot, [vec_mats[i] for i in idxs])
            for k in range(1, clifford_dim + 1)
            for idxs in combinations(range(len(vec_mats)), r=k)
        ],
        dot=lambda a, b: np.trace(a.conj().T @ b),
    )

    return mat_basis


def make_convert_vec(vec_mats):
    c_basis = make_mv_vec_basis(len(vec_mats))
    m_basis = make_cl_vec_basis(vec_mats)

    return ConvertVecBasis([c_basis, m_basis], ["c", "m"])
