from .deriv_symbol import DerivSymbol
from .symbol import Symbol

MAX_ONE_LINE_ELEM = 4


def symbol_sort_key(symbol):
    match symbol:
        case Symbol(name, is_conjugate):
            key = (0, name, is_conjugate)
        case DerivSymbol(symbol, _, derivativates):
            key = (1, bool(derivativates), symbol.name)  # TODO: more specific order?

    return key


try:
    # the library pretty-prints colorful.ColorfulString
    # here we register a pretty printer so that colors are printed

    def color_print(obj, printer, is_cycle):
        if is_cycle:
            printer.text("(cycle)")

        printer.text(str(obj))

    import colorful
    from IPython import get_ipython

    colorful.use_true_colors()

    text_printer = get_ipython().display_formatter.formatters["text/plain"]
    text_printer.for_type(colorful.core.ColorfulString, color_print)

except ImportError:
    pass
