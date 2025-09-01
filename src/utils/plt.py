from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable, Literal, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


Theme = Literal['base', 'light', 'dark', 'transparent'] | Mapping[str, str]
ThemeType = Literal['light', 'dark', 'indeterminate']


@contextmanager
def use_theme(*themes: Theme):
    with mpl.rc_context():
        stylesheet_dir = Path(__file__).parent / 'mplstyles'
        for theme in themes:
            if isinstance(theme, Mapping):
                plt.style.use(dict(theme))
            else:
                plt.style.use(stylesheet_dir / f'{theme}.mplstyle')
        yield


def autoclose(factory: Callable[..., Figure | None]) -> Callable[..., Figure | None]:
    @wraps(factory)
    def _autoclose(*args, **kwargs) -> Figure | None:
        fig = factory(*args, **kwargs)
        plt.close(fig)
        return fig

    return _autoclose
