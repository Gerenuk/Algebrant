from abc import abstractmethod
from collections.abc import MutableSequence
from typing import TYPE_CHECKING, LiteralString

import colorful

if TYPE_CHECKING:
    from IPython.lib.pretty import RepresentationPrinter


class ReprPrinter:
    """
    drop-in for ipython printer to be used to implement simpler __repr__
    """

    def __init__(self) -> None:
        self.result_parts: MutableSequence[str] = []

    def value(self):
        return "".join(self.result_parts)

    def text(self, text: str):
        self.result_parts.append(text)

    def pretty(self, obj: object):
        if isinstance(obj, colorful.core.ColorfulString):
            self.result_parts.append(obj.orig_string)
        else:
            self.result_parts.append(str(obj))  # TODO: will use repr for complex()?!

    def begin_group(self, _num: int, text: str):
        self.text(text)

    def end_group(self, _num: int, text: str):
        self.text(text)

    def break_(self):
        pass


class PlainReprMixin:
    @abstractmethod
    def _repr_pretty_(self, printer: "RepresentationPrinter", cycle: bool) -> None:
        pass

    def __repr__(self) -> LiteralString:
        printer = ReprPrinter()
        self._repr_pretty_(printer, cycle=False)
        return printer.value()
