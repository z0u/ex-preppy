import secrets
import time
from base64 import b64decode
from html import escape
from inspect import signature
from pathlib import Path
from typing import Generic, Protocol, Sequence, TypeVar, cast, override
from urllib.parse import quote

from airium import Airium
from IPython.core.formatters import DisplayFormatter
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import HTML, DisplayHandle, display
from matplotlib.figure import Figure

from utils.plt import Stylesheet, Theme, autoclose, use_theme

R = TypeVar('R', covariant=True)


class Displayer:
    """
    Display-or-update data in Jupyter notebooks.

    Like Jupyter's `display` function, but updates the displayed content
    in-place instead of creating a new output cell each time.
    """

    handle: DisplayHandle | None

    def __init__(self):
        self.handle = None

    def __call__(self, ob, **display_kwargs):
        """Display or update the given object in the notebook."""
        if not self.handle:
            self.handle = display(ob, display_id=True, **display_kwargs)
            # Sleeping here seems to resolve some weird race condition :shrug:
            # Without this, a single call followed immediately by a context
            # manager exit results in nothing being displayed.
            time.sleep(0.5)
        else:
            self.handle.update(ob, **display_kwargs)

    # DisplayHandle can't be pickled, so exclude it when serializing.
    def __reduce__(self):
        return (self.__class__, ())


class ImageFactory(Protocol, Generic[R]):
    def __call__(self) -> R: ...


class ThemeAwareImageFactory(Protocol, Generic[R]):
    def __call__(self, /, theme: Theme) -> R: ...


class ImageFactoryDisplayer[R]:
    """Displays images in Jupyter notebooks and saves them to a file."""

    path: Path
    alt_text: str | None

    _show: Displayer
    _last_data: str | None = None
    _factory: ImageFactory[R] | ThemeAwareImageFactory[R] | None

    def __init__(
        self,
        displayer: Displayer,
        path: str | Path,
        *,
        alt_text: str | None = None,
        live_theme: Sequence[Stylesheet] | None = None,
        light_theme: Sequence[Stylesheet] | None = None,
        dark_theme: Sequence[Stylesheet] | None = None,
    ):
        self.path = Path(path)
        if not self.path.suffix == '.png':
            raise NotImplementedError('Only png is supported.')
        self.alt_text = alt_text

        # Ordering here is important: the first one will be the default.
        variants: dict[str, tuple[str, Sequence[Stylesheet]]] = {}
        if light_theme:
            variants['light'] = ('(prefers-color-scheme: light)', light_theme)
        if dark_theme:
            variants['dark'] = ('(prefers-color-scheme: dark)', dark_theme)
        if live_theme:
            variants['live'] = ('', live_theme)

        if not variants:
            raise ValueError('No theme provided')

        self.variants = variants

        self._show = displayer
        self._factory = None

    def __call__(self, factory: ImageFactory[R] | ThemeAwareImageFactory[R]):
        self._factory = factory
        if 'live' in self.variants:
            data, metadata = self.render(self.variants['live'][1])
            self._show(data, metadata=metadata, raw=True)

    def render(self, themes: Sequence[Stylesheet]):
        formatter = cast(DisplayFormatter, InteractiveShell.instance().display_formatter)
        ob = None
        if self._factory is not None:
            sig = signature(self._factory)
            if 'theme' in sig.parameters:
                theme = Theme('light' if 'light' in themes else 'dark' if 'dark' in themes else 'indeterminate')
                ob = cast(ThemeAwareImageFactory[R], self._factory)(theme=theme)
            else:
                ob = cast(ImageFactory[R], self._factory)()
        data, metadata = formatter.format(ob, include=('image/png',))
        data['text/plain'] = f'{self.alt_text or "Image"}'
        return data, metadata

    def externalize(self):
        """
        Move the last displayed image to a file and display it as HTML.

        Reduces the size of notebooks.
        """
        if not self._factory:
            return

        variants = dict(self.variants)
        if 'live' in variants and len(variants) > 1:
            del variants['live']

        cache_buster = secrets.token_urlsafe()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        a = Airium()  # HTML builder
        with a.picture():
            for i, (name, (media, themes)) in reversed(list(enumerate(variants.items()))):
                data, _ = self.render(themes)
                if i > 0:
                    path = self.path.with_stem(f'{self.path.stem}.{name}')
                else:
                    path = self.path

                with open(path, 'wb') as f:
                    f.write(b64decode(data['image/png']))

                safe_src = f'{quote(path.as_posix())}?v={cache_buster}'
                a.source(srcset=safe_src, media=escape(media))
                if i == 0:
                    a.img(src=safe_src, alt=escape(data['text/plain']))

        formatter = cast(DisplayFormatter, InteractiveShell.instance().display_formatter)
        data, metadata = formatter.format(HTML(str(a)), include=('text/html'))
        data['text/plain'] = self.alt_text or 'Image'
        self._show(data, metadata=metadata, raw=True)


class MplFactoryDisplayer(ImageFactoryDisplayer[Figure | None]):
    def __init__(
        self,
        displayer: Displayer,
        path: str | Path,
        *,
        alt_text: str | None = None,
        live_theme: Sequence[Stylesheet] | None = None,
        light_theme: Sequence[Stylesheet] | None = None,
        dark_theme: Sequence[Stylesheet] | None = None,
        autoclose: bool = True,
    ):
        super().__init__(
            displayer,
            path,
            alt_text=alt_text,
            live_theme=live_theme,
            light_theme=light_theme,
            dark_theme=dark_theme,
        )
        self.autoclose = autoclose

    @override
    def __call__(self, factory: ImageFactory[Figure | None] | ThemeAwareImageFactory[Figure | None]):
        if self.autoclose:
            factory = autoclose(factory)
        super().__call__(factory)

    @override
    def render(self, themes: Sequence[Stylesheet]):
        with use_theme(*themes):
            return super().render(themes)
