from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd


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

grays2 = {
    'black': 0.0,
    'white': 1.0,
}

grays3 = {
    'black': 0.0,
    'gray': 0.5,
    'white': 1.0,
}

grays5 = {
    'black': 0.0,
    'dark gray': 0.25,
    'gray': 0.5,
    'light gray': 0.75,
    'white': 1.0,
}


nearest_xkcd = {
    # Primaries
    'red': 'bright red',
    'green': 'bright green',
    'blue': 'primary blue',
    # Secondaries
    'yellow': 'yellow',
    'cyan': 'cyan',
    'magenta': 'bright magenta',
    # Ternaries
    'orange': 'bright orange',
    'lime': 'lime green',
    'teal': 'turquoise green',
    'azure': 'deep sky blue',
    'purple': 'purplish blue',
    'pink': 'hot pink',
    # Grays
    'black': 'black',
    'dark gray': 'dark gray',
    'gray': 'medium gray',
    'light gray': 'light gray',
    'white': 'white',
}
"""
Mapping from our named colors to the nearest reasonable one from the XKCD color survey.
https://xkcd.com/color/rgb
"""


def get_named_colors_df(n_hues: Literal[3, 6, 12] = 12, n_grays: Literal[2, 3, 5] = 5) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for name, hue in (hues12 if n_hues == 12 else hues6 if n_hues == 6 else hues3).items():
        rows.append({'name': name, 'hsv': (hue, 1.0, 1.0)})
    for name, value in (grays5 if n_grays == 5 else grays3 if n_grays == 3 else grays2).items():
        rows.append({'name': name, 'hsv': (0.0, 0.0, value)})

    df = pd.DataFrame(rows)
    df['rgb'] = df['hsv'].apply(lambda hsv: colorsys.hsv_to_rgb(*hsv))
    return df
