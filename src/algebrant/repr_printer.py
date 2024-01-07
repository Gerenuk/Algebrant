import colorful


class ReprPrinter:
    """
    drop-in for ipython printer to be used to implement simpler __repr__
    """

    def __init__(self):
        self.result_parts = []

    def value(self):
        return "".join(self.result_parts)

    def text(self, text):
        self.result_parts.append(text)

    def pretty(self, obj):
        if isinstance(obj, colorful.core.ColorfulString):
            self.result_parts.append(obj.orig_string)
        else:
            self.result_parts.append(str(obj))

    def break_(self):
        pass
