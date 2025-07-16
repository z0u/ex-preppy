import html
import secrets
import urllib.parse
from base64 import b64decode
from pathlib import Path
from typing import cast

from IPython.core.formatters import DisplayFormatter
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import HTML, DisplayHandle, display


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
        else:
            self.handle.update(ob, **display_kwargs)

    # DisplayHandle can't be pickled, so exclude it when serializing.
    def __reduce__(self):
        return (self.__class__, ())


class ImageDisplayer:
    """Displays images in Jupyter notebooks and saves them to a file."""

    filepath: Path
    alt_text: str | None
    max_width: str | None

    _show: Displayer
    _last_data: str | None = None

    def __init__(
        self,
        displayer: Displayer,
        filepath: str | Path,
        *,
        alt_text: str | None = None,
        max_width: str | None = '70rem',
    ):
        self.filepath = Path(filepath)
        self.alt_text = alt_text
        self.max_width = max_width
        self._show = displayer

    def __call__(self, ob):
        formatter = cast(DisplayFormatter, InteractiveShell.instance().display_formatter)
        data, metadata = formatter.format(ob, include=('image/png',))
        data['text/plain'] = f'{self.alt_text or "Image"}'
        self._show(data, metadata=metadata, raw=True)
        self._last_data = cast(str, data.get('image/png'))

    def externalize(self):
        """
        Move the last displayed image to a file and display it as HTML.

        Reduces the size of notebooks.
        """
        if not self._last_data:
            return

        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'wb') as f:
            f.write(b64decode(self._last_data))

        alt_text = self.alt_text or ''

        style = f'max-width: {self.max_width};' if self.max_width is not None else ''

        escaped_alt = html.escape(alt_text)
        cache_buster = secrets.token_urlsafe()
        safe_src = urllib.parse.quote(self.filepath.as_posix())
        escaped_style = html.escape(style)
        tag = f'<img src="{safe_src}?v={cache_buster}" alt="{escaped_alt}" style="{escaped_style}" />'

        formatter = cast(DisplayFormatter, InteractiveShell.instance().display_formatter)
        data, metadata = formatter.format(HTML(tag), include=('text/html'))
        data['text/plain'] = self.alt_text or 'Image'
        self._show(data, metadata=metadata, raw=True)
