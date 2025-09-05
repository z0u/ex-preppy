primaries = {
    'red': 0 / 360,
    'green': 120 / 360,
    'blue': 240 / 360,
}
secondaries = {
    'yellow': 60 / 360,
    'cyan': 180 / 360,
    'magenta': 300 / 360,
}
ternaries = {
    'orange': 30 / 360,
    'lime': 90 / 360,  # Lime-green / yellow-green / neon-green.
    'teal': 150 / 360,
    'azure': 210 / 360,
    'purple': 270 / 360,
    'pink': 330 / 360,  # Actually _hot_ pink / neon-pink, not pastel pink.
}

hues3 = primaries
"""Trichromatic colors: red, green, blue (primaries)"""

hues6 = dict(sorted(((k, v) for k, v in (primaries | secondaries).items()), key=lambda item: item[1]))
"""Hexachromatic colors: red, yellow, etc. (primaries & secondaries)"""

hues12 = dict(sorted(((k, v) for k, v in (primaries | secondaries | ternaries).items()), key=lambda item: item[1]))
"""Dodecachromatic colors: red, orange, yellow, etc. (primaries, secondaries, & ternaries)"""

hues = hues12
