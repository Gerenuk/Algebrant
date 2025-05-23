from collections.abc import Iterable, Sequence
from dataclasses import field
from typing import TypeGuard, TypeVar

T = TypeVar("T")


def all_not_none(elems: Iterable[T | None]) -> TypeGuard[Sequence[T]]:
    return all(elem is not None for elem in elems)


def calculated_field(**kwargs):
    params = dict(init=False, hash=False, repr=False, compare=False)
    params.update(kwargs)
    return field(**params)
