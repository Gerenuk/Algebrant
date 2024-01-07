import numpy as np


def is_zero(val):
    if isinstance(val, np.ndarray):
        return np.all(val == 0)

    try:
        return val == 0
    except ValueError:
        return False


def is_identity(val):
    if isinstance(val, np.ndarray) and len(val.shape) == 2 and val.shape[0] == val.shape[1]:
        return np.all(val == np.identity(val.shape[0]))

    try:
        return val == 1 or abs(val - 1) < 1e-12
    except (ValueError, TypeError):
        return False


def conjugate(val):
    if hasattr(val, "conjugate"):
        return val.conjugate()

    return val
