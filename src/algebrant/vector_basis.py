import itertools

import numpy as np

from .format_number import format_vec_coef


class VecBasis:
    def __init__(self, basis_vecs, dot):
        """
        expects orthogonal basis_vecs wrt dot product
        """
        # check orthogonality
        self.dot = dot

        for (i_a, a), (i_b, b) in itertools.combinations(enumerate(basis_vecs), 2):
            prod = self.dot(a, b)
            if prod != 0:
                raise ValueError(
                    f"""Non-orthogonal elements:
vec[{i_a:2}] = {a}
vec[{i_b:2}] = {b}
dot     = {prod}
"""
                )

        self.basis_vecs = basis_vecs
        self.scalings = [self.dot(a, a) for a in basis_vecs]

        if any(s == 0 for s in self.scalings):
            zero_scaling_vectors = [v for s, v in zip(self.scalings, self.basis_vecs) if s == 0]
            raise ValueError(f"Zero length for vectors: {zero_scaling_vectors}")

    def __repr__(self):
        return f"Basis({len(self.basis_vecs)} dim.)"

    def __iter__(self):
        return iter(self.basis_vecs)

    def to_vec(self, elem, verify_complete=True):
        """
        returns as components in basis
        """
        result = []
        for basis, scaling in zip(self.basis_vecs, self.scalings):
            result.append(self.dot(basis, elem) / scaling)

        if verify_complete:
            self._verify_equal(elem, result)  # only when __eq__ yields boolean

        return np.array(result)

    def to_elem(self, vec):
        if not len(vec) == len(self.basis_vecs):
            raise ValueError(f"Incorrect vector length: expected {len(self.basis_vecs)}, received {len(vec)}")

        return sum(coef * base for coef, base in zip(vec, self.basis_vecs))

    def matrix(self, func):
        """
        Matrix elements assuming func is linear
        """
        result = []

        for basis_vec in self.basis_vecs:
            result.append(self.to_vec(func(basis_vec)))

        return np.array(result).T

    def _verify_equal(self, vec, coefs):
        result_vec = sum([coef * base for coef, base in zip(coefs, self.basis_vecs)])

        if isinstance(result_vec, np.ndarray):
            if np.issubdtype(result_vec.dtype, np.number):
                compare = np.all(np.isclose(result_vec, vec))
            else:
                compare = np.all(result_vec == vec)
        else:
            compare = result_vec == vec  # TODO: tolerant small deviations? use abs()?

        if not compare:
            error = " +\n".join(f"{coef} * {basis}" for coef, basis in zip(coefs, self.basis_vecs) if coef != 0)

            raise ValueError(
                f"""Base insufficient:
Result:   {result_vec}

Original: {vec}

Diff.:    {result_vec - vec}

Error.:    {error}
"""
            )

    def new_basis_matrix(self, new_basis):
        """
        use to_vec(x) @ new_basis_matrix(new_basis) to transform to new basis
        """
        return np.linalg.inv(np.array([self.to_vec(b) for b in new_basis]))

    def trace(self, func):
        result = 0

        for basis_vec in self.basis_vecs:
            result += self.dot(basis_vec, func(basis_vec))

        return result


class VecText:
    def __init__(self, names) -> None:
        self.names = names

    def to_names(self, vec):
        if len(vec) != len(self.names):
            raise ValueError(f"Vector has {len(vec)} elements, but expected {len(self.names)}: {vec}")

        return {name: coef for name, coef in zip(self.names, vec) if coef != 0}

    def to_vec(self, names):
        result = np.zeros(len(self.names))
        for name, val in names.items():
            result[self.names.index(name)] = val
        return result

    def to_str(self, vec):
        return " + ".join(f"{format_vec_coef(coef)}{name}" for name, coef in self(vec).items())
