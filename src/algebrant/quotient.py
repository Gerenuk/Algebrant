from .common import conjugate

"""
Assumes denominator is commutative

Currently quite rudimentary and mainly only for reporting quotient results

TODO:
* Cl / Cl makes numer {1:Cl} instead of spreading
* implement addition (with quotient or anything else)
* implement cancel Basis
* implement __eq__, __add__, __sub__, __rtruediv__
* simplification when scalars involved
* A/A -> 1
"""


QUOTIENT_OP_PRIO = -1


class Quotient:
    def __init__(self, numer, denom=1, *, op_prio=QUOTIENT_OP_PRIO):
        if numer == 0:
            denom = 1

        self.numer = numer
        self.denom = denom
        self.op_prio = op_prio

    def _create(self, numer, denom):
        return self.__class__(numer, denom, op_prio=self.op_prio)

    def __mul__(self, other):
        if hasattr(other, "op_prio") and other.op_prio < self.op_prio:
            return NotImplemented

        new_numer = self.numer * other
        if new_numer == self.denom:
            return 1

        return self._create(new_numer, self.denom)

    def __rmul__(self, other):
        if hasattr(other, "op_prio") and other.op_prio < self.op_prio:
            return NotImplemented

        new_numer = other * self.numer

        if new_numer == self.denom:
            return 1

        return self._create(new_numer, self.denom)

    def __neg__(self):
        return self._create(-self.numer, self.denom)

    def __repr__(self):
        return f"({self.numer} / {self.denom})"

    def __iter__(self):
        return iter((self.numer, self.denom))

    def __eq__(self, other):
        if other == 0:
            return self.numer == 0

        if isinstance(other, Quotient):
            return self.numer * other.denom == other.numer * self.denom

        raise NotImplementedError("Equality not implemented")

    def conjugate(self):
        return Quotient(conjugate(self.numer), conjugate(self.denom), op_prio=self.op_prio)

    def _repr_pretty_(self, printer, cycle):
        if cycle:
            return printer.text("...")

        printer.pretty(self.numer)
        printer.break_()
        printer.text("───")
        printer.break_()
        printer.pretty(self.denom)
