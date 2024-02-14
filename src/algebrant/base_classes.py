from abc import ABC, abstractmethod, abstractproperty
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

    @abstractmethod
    def conjugate(self):
        raise NotImplementedError()


class BaseBasis(ABC):
    """
    basis for algebra with multiplication
    """

    def mul(self, factor1, basis2, factor2):
        """
        used by Algebra multiplication and needed factor2 has no-trivial change when going through basis1
        """
        prod_factor = factor1 * factor2
        return {basis: prod_factor * factor for basis, factor in (self * basis2).items()}

    @abstractmethod
    def conjugate(self, factor) -> tuple():  # needs this name because apparently someone checks for "conjugate" name
        raise NotImplementedError()

    @abstractproperty
    def scalar_part(self):
        raise NotImplementedError()

    @abstractmethod
    def __mul__(self, other) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def _sort_key(self):  # for sorting repr
        raise NotImplementedError()

    def inverse(self) -> dict:
        """
        Algebra can invert if only a single Basis in the sum
        """
        return NotImplemented
