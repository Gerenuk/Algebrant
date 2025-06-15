from typing import Counter

import numpy as np

# from IPython import get_ipython

"""
TODO:
"""


# ipython = get_ipython()


# def repr_coef(x):
#    return ipython.display_formatter.format(x)[0]["text/plain"]


class VecBasis:
    def __init__(self, basis_vecs, *, dot, names=None, min_abs=1e-8) -> None:
        """
        basis_vecs need only scalar multiplication and the provided dot function
        dot non-degenerate, but not necessarily symmetric (e.g. lambda a, b: a @ T @ b)
        """
        self.basis_vecs = basis_vecs
        self._dot = dot
        self.dim = len(basis_vecs)
        self.min_abs = min_abs

        if names is not None:
            if not len(basis_vecs) == len(names):
                raise ValueError(
                    f"Incorrect length names: len(basis_vecs)={len(basis_vecs)}, len(names)={len(names)}"
                )

            if len(set(names)) != len(names):
                raise ValueError(
                    f"Duplicate names: { {name: count for name, count in Counter(names).items() if count > 1} }"
                )

            self.names = names
        else:
            self.names = [f"e{i}" for i in range(1, len(basis_vecs) + 1)]

        self.gram_matrix = np.array(
            [[dot(b1, b2) for b2 in basis_vecs] for b1 in basis_vecs]
        )

        # TODO: hack since linalg cannot for float128
        if self.gram_matrix.dtype == np.float128:
            self.gram_matrix = self.gram_matrix.astype(np.float64)

        rank = np.linalg.matrix_rank(self.gram_matrix)
        if rank != self.gram_matrix.shape[0]:
            raise ValueError(f"Basis is rank-deficient: {rank} < {len(basis_vecs)}")

        self.inv_gram_matrix = np.linalg.inv(self.gram_matrix)

        self.dual_basis_vecs = [  # currently used only for .trace()
            sum(
                coef * vec
                for coef, vec in zip(
                    self.inv_gram_matrix[i, :], self.basis_vecs, strict=True
                )
            )
            for i in range(self.dim)
        ]

    def dot(self, vec1, vec2):
        return self._dot(vec1, vec2)

    def to_coef(self, vec, *, verify=True):
        """
        basis coefficients
        such that x == V.to_vec(V.to_coef(x))
        """
        coef0 = np.array([self._dot(b, vec) for b in self.basis_vecs])

        coefs = self.inv_gram_matrix @ coef0

        if verify:
            elem_from_coef = self.to_vec(coefs)

            abs_val = None

            if isinstance(elem_from_coef, np.ndarray):
                is_equal = np.all(np.isclose(elem_from_coef, vec))
            else:
                try:
                    abs_val = abs(elem_from_coef - vec)
                    is_equal = abs_val < self.min_abs
                except TypeError:
                    is_equal = elem_from_coef == vec

            # TODO: use (vector) norm instead (for imprecise comparison)

            if not isinstance(is_equal, (bool, np.bool_)):
                raise ValueError(f"Type {type(is_equal)} of comparison is not bool")

            if not is_equal:
                raise ValueError(
                    f"Missing basis for coefs {coefs}:\ndiff"
                    + (f" abs {abs_val}" if abs_val is not None else "")
                    + f" =\n{vec - elem_from_coef}\n=\n {vec} (orig)\n-\n{elem_from_coef} (calc)"
                )

        return coefs

    def to_dict(self, vec, *, verify=True):
        coefs = self.to_coef(vec, verify=verify)

        return {
            name: coef
            for name, coef in sorted(zip(self.names, coefs))
            if coef != 0 and (not hasattr(coef, "__abs__") or abs(coef) >= self.min_abs)
        }

    def to_str(self, vec, *, verify=True, coef_formatter=None):
        if coef_formatter is None:
            coef_formatter = str

        coef_dict = self.to_dict(vec, verify=verify)

        result = " + ".join(
            (coef_formatter(coef) + " " if coef != 1 else "") + name
            for name, coef in coef_dict.items()
        )

        if not result:
            result = "0"

        return result

    def to_vec(self, coefs):
        if len(coefs) != len(self.basis_vecs):
            raise ValueError(
                f"Number of coefficients {len(coefs)} does not equal dimension of basis {len(self.basis_vecs)}"
            )

        return sum(coef * b for coef, b in zip(coefs, self.basis_vecs, strict=True))

    def to_matrix(self, op):
        """
        matrix of a bilinear operator
        such that V.to_coef(op(x)) == V.to_matrix(op) @ V.to_coef(x)
        therefore a matrix in the coefficient basis
        """
        return np.array([self.to_coef(op(b)) for b in self.basis_vecs]).T

    # def dual(self, vec):
    #    dual_coefs = self.inv_gram_matrix @ self.to_coef(vec)
    #    return sum(coef * vec for coef, vec in zip(dual_coefs, self.dual_basis_vecs, strict=True))

    def trace(self, op):
        """
        trace
        such that V.trace(lambda x: m @ x) == np.trace(m)
        """
        return sum(
            self._dot(b_d, op(b))
            for b_d, b in zip(self.dual_basis_vecs, self.basis_vecs, strict=True)
        )

    def __repr__(self):
        return f"VecBasis(dim={self.dim})"

    def __iter__(self):
        return iter(self.basis_vecs)


class ConvertVecBasis:
    def __init__(self, *vec_basis_list):
        dims = set(v.dim for v in vec_basis_list)
        if len(dims) != 1:
            raise ValueError(
                f"Inconsistent dimensions {[v.dim for v in vec_basis_list]}"
            )

        self.dim = next(iter(dims))  # used only for __repr__

        self.vec_basis_list = vec_basis_list

    def __call__(self, elem, idx_from, idx_to, verify=True):
        """
        idx = None uses a raw coefficient vector
        """
        if idx_from is not None:
            coefs = self.vec_basis_list[idx_from].to_coef(elem, verify=verify)
        else:
            coefs = elem

        if idx_to is not None:
            result = self.vec_basis_list[idx_to].to_vec(coefs)
        else:
            result = coefs

        return result

    def __repr__(self):
        return f"ConvertVecBasis(dim={self.dim})"
