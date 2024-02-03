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
    def conjugate(self, factor) -> tuple():
        pass

    @abstractproperty
    def scalar_part(self):
        pass

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
