import itertools


def conjugate(value):
    if isinstance(value, (int, float)):
        return value
    elif hasattr(value, "conjugate"):
        return value.conjugate()
    else:
        raise TypeError(f"Unsupported type for conjugation: {type(value)}")


def cl_dot(a, b):
    factors_a = a.basis_factor.to_dict
    factors_b = b.basis_factor.to_dict
    return sum(
        conjugate(factors_a[basis]) * factors_b[basis]
        for basis in factors_a.keys() & factors_b.keys()
    )


def sqr_to_scalar(val: "CliffordAlgebra"):
    """
    Returns (scalar**2, non_scalar**2) but non_scalar only if it squares to a scalar
    """
    non_scalar = val.take_grades(lambda g: g > 0)

    bases, values = zip(*non_scalar)
    all_anti_commute = not any(
        b1.commutes_with(b2) for b1, b2 in itertools.combinations(bases, r=2)
    )

    non_scalar_sqr = (
        sum(value**2 * base.sqr() for base, value in zip(bases, values))
        if all_anti_commute
        else None
    )

    return val.scalar_part**2, non_scalar_sqr


def mv_sqrt(A):
    """
    TODO: sometimes numerical issues; better reverse
    """
    from cmath import sqrt

    ns = A.take_grades(lambda g: g > 0)

    scalar_sqr, non_scalar_sqr = sqr_to_scalar(A)

    if non_scalar_sqr is None:
        raise NotImplementedError(f"Cannot take sqrt of {A}")

    scalar = A.scalar_part

    if non_scalar_sqr == 0:
        c1 = sqrt(scalar)
        c2 = -c1
    else:
        b = sqrt(scalar_sqr - non_scalar_sqr)
        c1 = sqrt((scalar + b) / 2)
        c2 = sqrt((scalar - b) / 2)

    return (
        c1 + ns / (2 * c1),
        c2 + ns / (2 * c2),
    )
