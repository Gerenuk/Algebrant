from collections.abc import Iterable
from typing import Protocol, Self


class GradedProtocol(Protocol):
    @property
    def grade(self) -> int | None: ...

    @property
    def is_odd(self) -> bool: ...

    @property
    def i(self) -> Iterable[tuple[Self, int]]: ...

    @property
    def r(self) -> Iterable[tuple[Self, int]]: ...

    @property
    def cl(self) -> Iterable[tuple[Self, int]]: ...
