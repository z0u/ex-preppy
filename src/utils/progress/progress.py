import asyncio
import time
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    TypeVar,
    cast,
    override,
)

from IPython.display import HTML

from utils.coro import debounced
from utils.nb import displayer
from utils.progress._progress import _Progress
from utils.progress.html import render_progress_bar
from utils.progress.iterators import AsyncIteratorWrapper
from utils.progress.model import BarData

T = TypeVar('T')


class Progress(_Progress, Generic[T], AsyncIterable[T]):
    """
    A simple, Jupyter-friendly progress bar using HTML and display updates.

    Avoids ipywidgets for better notebook saving and compatibility.
    Designed to be theme-agnostic (light/dark).
    Supports suffix messages and a dictionary of metrics.
    Can be used as a context manager or an iterator source.
    """

    _show: Callable[[Any], None]
    _iterator: Iterator[T] | AsyncIterator[T]
    _draw_task: asyncio.Task

    def __init__(
        self,
        items: Iterable[T] | AsyncIterable[T] | None = None,
        total: int | None = None,
        description: str = '',
        initial_metrics: dict[str, Any] | None = None,
        interval: float = 0.05,  # Min time between updates
        auto_yield: bool = False,
    ):
        if total is not None and total < 0:
            raise ValueError('total must be non-negative')

        if items is not None:
            if total is None:
                try:
                    total = len(items)  # type: ignore
                except (TypeError, AttributeError) as e:
                    raise ValueError('Cannot determine total length of iterable, please provide `total`.') from e
            if isinstance(items, (AsyncIterator, AsyncIterable)):
                self._iterator = aiter(items)
            else:
                self._iterator = iter(items)
        elif total is not None:
            # When iterable is None, we know T must be int.
            self._iterator = cast(Iterator[T], iter(range(total)))
        else:
            raise ValueError('Must provide either `iterable` or `total`.')

        self._auto_yield = auto_yield

        self._show = displayer()
        self._debounced_draw = debounced(interval=interval)(lambda: self._draw())
        self._draw_task = asyncio.Task(noop())

        # Setting these triggers a draw; see _Progress
        self.total = total
        self.description = description
        self.metrics = initial_metrics or {}
        self.suffix = ''
        self.start_time = time.monotonic()
        self.count = 0

    @override
    def _on_change(self):
        self._display('debounced')

    def _display(self, mode: Literal['immediate', 'debounced']):
        if mode == 'immediate':
            self._draw_task.cancel()
            self._draw()
        else:
            self._draw_task = self._debounced_draw()

    def _draw(self):
        html_content = self._render_html()
        self._show(HTML(html_content))

    def _render_html(self) -> str:
        """Generate the HTML for the progress bar and metrics grid."""
        data = BarData(
            count=self.count,
            total=self.total,
            description=self.description,
            suffix=self.suffix,
            elapsed_time=time.monotonic() - self.start_time,
        )
        return render_progress_bar(data, self.metrics)

    @override
    def close(self) -> None:
        """Ensuring the bar is drawn one last time."""
        self._display('immediate')

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            raise

    def __aiter__(self) -> AsyncIterator[T]:
        self.count = 0
        self.start_time = time.monotonic()
        self._display('debounced')

        return AsyncIteratorWrapper(self, self._iterator, self._auto_yield)


async def noop():
    pass
