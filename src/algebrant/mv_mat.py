from functools import reduce
from itertools import combinations
from math import prod

import numpy as np

from algebrant.clifford import cl_dot

from .common import conjugate, scalar_part
from .creation import E
from .vector_basis import ConvertVecBasis, VecBasis, mat_dot

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


def grade_1_to_all(es, *, create_func=prod):
    return [create_func(grade_es) for grade in range(len(es) + 1) for grade_es in combinations(es, r=grade)]


def make_mv_vec_basis_from_dim(dim, start_idx=1):
    return VecBasis(
        grade_1_to_all(range(start_idx, start_idx + dim), create_func=lambda x: E(*x)),
        dot=cl_dot,
    )


def make_mv_vec_basis_from_vec(basis_vecs):
    return VecBasis(
        grade_1_to_all(basis_vecs, create_func=lambda x: prod(x) if x else E()),
        dot=cl_dot,
    )


def make_mat_gen_vec_basis(vec_mats):
    mat_dim = vec_mats[0].shape[0]
    identity = np.identity(mat_dim)

    mat_basis = VecBasis(
        grade_1_to_all(vec_mats, create_func=lambda mats: reduce(np.dot, mats) if mats else identity),
        dot=mat_dot,
    )

    return mat_basis


def make_convert_vec(vec_mats):
    c_basis = make_mv_vec_basis_from_dim(len(vec_mats))
    m_basis = make_mat_gen_vec_basis(vec_mats)

    return ConvertVecBasis([c_basis, m_basis], ["c", "m"])


class ClMat:
    def __init__(self, vec_mats):
        self.convert = make_convert_vec(vec_mats)

    def to_mat(self, elem):
        return self.convert(elem, "c", "m")

    def to_cl(self, elem):
        return self.convert(elem, "m", "c")

    @classmethod
    def from_dim(cls, dim):
        mats = make_cl_mat_basis(dim)
        return cls(mats)

    @classmethod
    def chiral(cls):
        mats = list(map(np.matrix, make_mat_basis_from_pauli([(3, 0), (2, 1), (2, 2), (2, 3)])))
        return cls(mats)
