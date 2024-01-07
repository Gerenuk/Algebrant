import itertools
import math
from collections import deque


def anti_comm(x, y):
    """
    what the rule should return when x y -> - y x
    """
    return (((y,), -1),), 2, (x,)


def comm(x, y):
    """
    what the rule should return when x y -> y x
    """
    return (((y,), 1),), 2, (x,)


def single(x, factor=1):
    return (((x,), factor),)


def _normalize_step(seq, rule):
    """
    seq: sequence whose starting elements should be checked with rule
    return: Dict[Tuple[Elem,...]: Factor]
    """
    seq = deque(seq)
    last_part = []
    result_parts = []

    modified = False

    while seq:
        # print(":", result_parts, last_part, seq)

        next_result = rule(seq)

        if next_result is None:  # no rule matched
            elem = seq.popleft()
            last_part.append(elem)
            continue

        modified = True

        if last_part:
            result_parts.append(((last_part, 1),))
            last_part = []

        if next_result == 0:  # shortcut for zero
            return {}

        next_part, num_consume, prepend = next_result

        if next_part is not None:
            result_parts.append(next_part)

        for _ in range(num_consume):
            seq.popleft()

        if prepend is not None:  # needs to be reversed order
            seq.extendleft(prepend)

    if not modified:
        return None

    if last_part:
        result_parts.append(((last_part, 1),))

    result = {}
    for basis_factors in itertools.product(*result_parts):
        basis = tuple(itertools.chain.from_iterable(basis for basis, _factor in basis_factors))
        factor = math.prod(factor for _basis, factor in basis_factors)

        if basis in result:
            result[basis] += factor
        else:
            result[basis] = factor

    return result


def normalize(basis_factors, rule):
    """
    rule(seq) should return:
    None if current seq does not start with elements to be changed
    0 if whole expression is zero
    TupleOfSum[ Tuple[TupleOfBasis[elem, ...], factor], ... ], num_consume, prepend_reversed]
    first=None if nothing to be added
    """
    result = {}  # will contain normalized result only

    unchecked_basis_factors = basis_factors

    while unchecked_basis_factors:
        new_unchecked_basis_factors = {}

        for basis, factor in unchecked_basis_factors.items():
            if basis in result:  # is already normalized
                result[basis] += factor
                continue

            result_step = _normalize_step(basis, rule)

            if result_step is None:  # basis turns out to be normalized
                result[basis] = factor
                continue

            # new result will need checking in next iteration
            for basis, factor2 in result_step.items():
                if basis in new_unchecked_basis_factors:
                    new_unchecked_basis_factors[basis] += factor * factor2
                else:
                    new_unchecked_basis_factors[basis] = factor * factor2

        unchecked_basis_factors = new_unchecked_basis_factors

    return result
