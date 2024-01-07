from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseSymbol(ABC):
    """
    symbols from which usually multiple are combined into a basis
    """

    def _repr_pretty_(self, printer, cycle):
        """
        For pretty-printing in Jupyter through IPython
        """
        raise NotImplementedError()

    def conjugate(self):
        raise NotImplementedError()


class BaseBasis(ABC):
    """
    basis for algebra with multiplication
    make sure to return Algebra.UNITY_BASIS when appropriate
    """

    @abstractmethod
    def __mul__(self, other) -> dict:
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __lt__(self, other):  # for sorting repr
        pass

    def inverse(self) -> dict:
        """
        Algebra can invert if only a single Basis in the sum
        """
        return NotImplemented
