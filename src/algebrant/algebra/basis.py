from typing import Protocol

BasisSortKey = tuple[tuple[int, ...], tuple[str, ...]]


class BasisProtocol(Protocol):
    """
    basis for algebra with multiplication
    """

    def __hash__(self) -> int: ...

    def __eq__(self, other) -> bool: ...

    @property
    def sort_key(self) -> BasisSortKey: ...

    @property
    def is_unity(self) -> bool: ...
