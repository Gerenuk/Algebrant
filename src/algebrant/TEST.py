from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Protocol, Self, Sequence, TypeVar


class P(Protocol):
    def xyz(self) -> None:
        pass


T = TypeVar("T")


class A[T: P]:
    def __init__(self, x: T) -> None:
        self.x = x

    def p(self) -> None:
        return self.x.xyz()
