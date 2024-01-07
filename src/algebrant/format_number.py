import numbers

try:
    import colorful

    colorful.use_true_colors()
    imag_col = colorful.goldenrod1
except ImportError:
    imag_col = lambda x: x


def _format_float(number):
    if number.is_integer():
        return str(int(number))
    return str(number)


def format_number(number, use_color=False, clip_value=1e-10):
    if use_color:
        _imag_col = imag_col
    else:
        _imag_col = lambda x: x

    if isinstance(number, complex):
        result = []
        real = number.real
        imag = number.imag
        abs_val = abs(number)

        if abs(real) < abs_val * clip_value:
            real = 0

        if abs(imag) < abs_val * clip_value:
            imag = 0

        if real != 0:
            result.append(_format_float(real))
        if imag != 0:
            if imag > 0 and result:
                result.append("+")
            if imag == 1:
                result.append(_imag_col("i"))
            elif imag == -1:
                result.append("-" + _imag_col("i"))
            else:
                result.append(f"{_format_float(imag)}" + _imag_col("i"))
        text = "".join(map(str, result)) if result else "0"
        if len(result) > 1:
            text = "(" + text + ")"
        return text
    elif isinstance(number, float):
        return _format_float(number)

    return str(number)


def format_vec_coef(coef):
    if coef == 1:
        text = ""
    elif coef == -1:
        text = "-"
    elif isinstance(coef, numbers.Number):
        text = f"{format_number(coef)} "
    else:
        text = f"{coef} "

    return text
