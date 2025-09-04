from ex_color.data.color_cube import color_axes


def prettify(value: float, tolerance=1e-10):
    """Convert a float to a string, attempting to make it more human-readable."""
    from sympy import nsimplify, pretty

    # result = fu(value)
    result = nsimplify(value, tolerance=tolerance, rational_conversion='exact')
    s = pretty(result, use_unicode=True)
    if '\n' in s:
        return str(result)
    else:
        return s


def axname(axis_char: str) -> str:
    name = color_axes.get(axis_char.lower(), axis_char)
    return name
