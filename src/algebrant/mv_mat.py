from functools import reduce
from itertools import combinations
from math import prod

import numpy as np

from algebrant.algebra.algebra import dot_product
from algebrant.clifford.clifford_basis import cl_dot

from ...archive.creation import E
from .vector_basis import ConvertVecBasis, VecBasis

Paulis = [
    np.array([[1, 0], [0, 1]]),  # identity
    np.array([[1, 0], [0, -1]]),  # sigma_z
    np.array([[0, 1], [1, 0]]),  # sigma_x
    np.array([[0, 1j], [-1j, 0]]),  # -sigma_y
]


def make_mats_from_paulis(indices):
    return [reduce(np.kron, [Paulis[i] for i in e_idx]) for e_idx in indices]


def grade_1_to_all(es, *, create_func=prod):
    return [
        create_func(grade_es)
        for grade in range(len(es) + 1)
        for grade_es in combinations(es, r=grade)
    ]


def make_anti_comm_mats(n: int):
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

    return make_mats_from_paulis(indices)


def make_cl_mats(dim: int):
    if not dim % 2 == 0:
        raise ValueError(
            "Only even dimensions supported (odd dimensions require a direct product of 2 matrices)"
        )

    n = dim // 2
    vec_mats = make_anti_comm_mats(n)[:dim]
    return vec_mats


def make_cl_vec_basis_from_dim(dim, start_idx=1):
    return VecBasis(
        grade_1_to_all(range(start_idx, start_idx + dim), create_func=lambda x: E(*x)),
        dot=cl_dot,
    )


def make_cl_vec_basis_from_vec(basis_vecs):
    return VecBasis(
        grade_1_to_all(basis_vecs, create_func=lambda x: prod(x) if x else E()),
        dot=cl_dot,
    )


def make_mat_vec_basis_from_mats(vec_mats):
    mat_dim = vec_mats[0].shape[0]
    identity = np.identity(mat_dim)

    mat_basis = VecBasis(
        grade_1_to_all(
            vec_mats,
            create_func=lambda mats: reduce(np.dot, mats) if mats else identity,
        ),
        dot=dot_product,
    )

    return mat_basis


def validate_cl_mats(cl_bases, mat_bases):
    # validate matrices
    for i, (base, mat) in enumerate(zip(cl_bases, mat_bases)):
        base_sqr = (base**2).scalar_part  # TODO
        mat_sqr = mat @ mat

        if not np.all(base_sqr * np.eye(mat.shape[0]) == mat_sqr):
            raise ValueError(
                f"At index {i}: {base}^2 = {base_sqr} must be equal to square of matrix {mat}^2 = {mat_sqr}"
            )

    for (i, mat1), (j, mat2) in combinations(enumerate(mat_bases), 2):
        if not np.all(mat1 @ mat2 + mat2 @ mat1 == 0):
            raise ValueError(
                f"At indices {i}, {j}: Matrix {mat1} does not anti-commute with {mat2}"
            )


class ClMat:
    def __init__(self, *, vec_mats, vec_cls):
        validate_cl_mats(vec_cls, vec_mats)

        self.dim = len(vec_mats)  # currently only for __repr__
        self.vec_mats = vec_mats  # only for reference of creation
        self.vec_cl = vec_cls

        c_basis = make_cl_vec_basis_from_vec(vec_cls)
        m_basis = make_mat_vec_basis_from_mats(vec_mats)
        self.convert = ConvertVecBasis(c_basis, m_basis)

    def to_mat(self, elem):
        return self.convert(elem, 0, 1)

    def to_cl(self, elem):
        return self.convert(elem, 1, 0)

    @classmethod
    def from_dim(cls, dim):
        mats = make_cl_mats(dim)
        return cls(mats, [E(i + 1) for i in range(dim)])

    def __repr__(self):
        return f"ClMat({self.dim})"
