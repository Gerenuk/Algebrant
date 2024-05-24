import numpy as np


class VecBasis:
    def __init__(self, basis_vecs, *, dot):
        self.basis_vecs = basis_vecs
        self.dot = dot
        self.dim = len(basis_vecs)

        trans = np.array([[dot(b2, b1) for b2 in basis_vecs] for b1 in basis_vecs])

        rank = np.linalg.matrix_rank(trans)
        if rank != trans.shape[0]:
            raise ValueError(f"Basis is rank-deficient: {rank} < {len(basis_vecs)}")

        self.inv_trans = np.linalg.inv(trans)

    def to_coef(self, elem, verify=True):
        vec = np.array([self.dot(b, elem) for b in self.basis_vecs])
        coefs = self.inv_trans @ vec

        if verify:
            elem_from_coef = self.to_vec(coefs)
            if isinstance(elem, np.ndarray):
                is_equal = np.all(np.isclose(elem_from_coef, elem))
            else:
                is_equal = elem_from_coef == elem

            # TODO: use (vector) norm instead (for imprecise comparison)

            if not is_equal:
                raise ValueError(f"Missing basis {elem - elem_from_coef}")

        return coefs

    def to_vec(self, coefs):
        if len(coefs) != len(self.basis_vecs):
            raise ValueError(
                f"Number of coefficients {len(coefs)} does not equal dimension of basis {len(self.basis_vecs)}"
            )

        return sum(coef * b for coef, b in zip(coefs, self.basis_vecs))

    def __repr__(self):
        return f"VecBasis(dim={self.dim})"

    def __iter__(self):
        return iter(self.basis_vecs)


class ConvertVecBasis:
    def __init__(self, vec_basis_list, names):
        dims = set(v.dim for v in vec_basis_list)
        if len(dims) != 1:
            raise ValueError(f"Inconsistent dimensions {[v.dim for v in vec_basis_list]}")

        self.dim = vec_basis_list[0].dim
        self.names = names
        self.vec_basis_list = vec_basis_list

    def __call__(self, elem, name_from, name_to, verify=True):
        if name_from != "v":
            coefs = self.vec_basis_list[self.names.index(name_from)].to_coef(elem, verify=verify)
        else:
            coefs = elem

        if name_to != "v":
            result = self.vec_basis_list[self.names.index(name_to)].to_vec(coefs)
        else:
            result = coefs

        return result

    def __repr__(self):
        return f"ConvertVecBasis({self.names}, dim={self.dim})"
