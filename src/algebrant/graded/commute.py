from collections.abc import Iterable

from algebrant.graded.graded_protocol import GradedProtocol

# @singledispatch
# def commute[Basis, Factor](basis: Basis, factor: Factor) -> Iterable[tuple[Basis, Factor]]:
#     raise NotImplementedError(f"commute not implemented for basis type {type(basis)}")


# @commute.register
def commute[Basis: GradedProtocol, Factor](
    basis: Basis, factor: Factor
) -> Iterable[tuple[Basis, Factor]]:
    if basis.is_odd:
        return [(basis, getattr(factor, "vector_conjugate", factor))]

    return [(basis, factor)]
