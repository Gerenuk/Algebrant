try:
    # the library pretty-prints colorful.ColorfulString
    # here we register a pretty printer so that colors are printed

    def color_print(obj, printer, is_cycle):
        if is_cycle:
            printer.text("(cycle)")

        printer.text(str(obj))

    import colorful
    from IPython.core.getipython import get_ipython

    colorful.use_true_colors()  # type: ignore

    text_printer = get_ipython().display_formatter.formatters["text/plain"]  # type: ignore
    text_printer.for_type(colorful.core.ColorfulString, color_print)

except ImportError:
    pass
