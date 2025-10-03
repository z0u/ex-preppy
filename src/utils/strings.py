_supermap = {
    '0': '⁰',
    '1': '¹',
    '2': '²',
    '3': '³',
    '4': '⁴',
    '5': '⁵',
    '6': '⁶',
    '7': '⁷',
    '8': '⁸',
    '9': '⁹',
    '+': '⁺',
    '-': '⁻',
    '.': '·',
}


def sup(num: float, *, precision: int = 2) -> str:
    """
    Convert a number to its superscript representation.

    This converts the entire number, not just the exponent.
    """
    if isinstance(num, int):
        num_str = str(num)
    else:
        num_str = f'{num:.{precision}:f}'.rstrip('0').rstrip('.')
    return ''.join(_supermap.get(c, c) for c in num_str)
