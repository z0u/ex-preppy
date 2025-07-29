import asyncio
import time
from typing import Any, AsyncIterable, AsyncIterator, Generic, Iterable, Iterator, TypeVar, cast, override

from utils.coro import debounced
from utils.nb import displayer, is_graphical_notebook
from utils.progress._progress import _Progress
from utils.progress.html import render_progress_bar
from utils.progress.iterators import async_iterator_wrapper, sync_iterator_wrapper
from utils.progress.model import BarData, Mark

T = TypeVar('T')


class ProgressBase(_Progress, Generic[T]):
    def __init__(
        self,
        total: int,
        description: str,
        initial_metrics: dict[str, Any],
    ):
        if total < 0:
            raise ValueError('total must be non-negative')

        if is_graphical_notebook():
            self._show = _DisplayInNotebook()
        else:
            self._show = _DisplayInConsole()

        # Setting these usually triggers a draw; see _Progress
        self._draw_on_change = False
        self.total = total
        self.description = description
        self.metrics = initial_metrics or {}
        self.markers = []
        self.suffix = ''
        self.start_time = time.monotonic()
        self.count = 0
        self._draw_on_change = True

    def update(
        self,
        total: int | None = None,
        count: int | None = None,
        description: str | None = None,
        metrics: dict[str, Any] | None = None,
        suffix: str | None = None,
    ):
        """
        Update the progress bar with new values.

        Usually this isn't necessary: you can just mutate the values directly, and the
        bar will redraw. But you might need to use this if you're using an API that
        needs a progress update function.
        """
        draw_was_enabled = self._draw_on_change
        self._draw_on_change = False
        try:
            if total is not None:
                self.total = total
            if count is not None:
                self.count = count
            if description is not None:
                self.description = description
            if metrics is not None:
                self.metrics = metrics
            if suffix is not None:
                self.suffix = suffix
        finally:
            self._draw_on_change = draw_was_enabled
        self._on_change()

    def mark(self, label: str, count: int | None = None):
        """
        Add a marker.

        Args:
            label: The label for the marker.
            count: The count at which to place the marker. If None, uses the current count.
        """
        self.markers.append(Mark(count if count is not None else self.count - 1, label))

    def print(self, *args, **kwargs):
        """Print without damaging the progress bar display."""
        self._show.print(*args, **kwargs)

    def before_print(self):
        """Clear the line so other content can be displayed"""
        self._show.clear_for_print()

    def after_print(self):
        self._debounced_draw()

    @override
    def _on_change(self):
        if self._draw_on_change:
            self._debounced_draw()

    def _debounced_draw(self) -> None:
        raise NotImplementedError

    def _draw(self):
        self._show.show(self)

    def __repr__(self) -> str:
        fraction = self.count / self.total if self.total > 0 else 1
        return f'{self.description}: {fraction:.1%} [{self.count:d}/{self.total:d}]'

    def _repr_html_(self) -> str:
        """Generate the HTML for the progress bar and metrics grid."""
        data = BarData(
            count=self.count,
            total=self.total,
            description=self.description,
            suffix=self.suffix,
            elapsed_time=time.monotonic() - self.start_time,
            markers=self.markers,
        )
        return render_progress_bar(data, self.metrics)

    @override
    def close(self) -> None:
        """Ensuring the bar is drawn one last time."""
        self._draw()
        self._show.finalize()


class Progress(ProgressBase, AsyncIterable[T]):
    """
    A simple, Jupyter-friendly progress bar using HTML and display updates.

    - Avoids ipywidgets for better notebook saving and compatibility.
    - Designed to be theme-agnostic (light/dark).
    - Supports suffix messages and a dictionary of metrics.
    - Can be used as a context manager or iterator source (`async with`, `async for`).

    This progress bar features debounced updates, with redraws ocurring at both the
    leading and trailing edges. Redraws execute in the asyncio event loop, so you must
    yield control inside your loop with a truly asynchronous operation, like
    `await asyncio.sleep(0)`. For convenience, you can wrap it in `co_op` to
    automatically do that once per loop.
    """

    def __init__(
        self,
        items: Iterable[T] | AsyncIterable[T] | None = None,
        total: int | None = None,
        description: str = '',
        initial_metrics: dict[str, Any] | None = None,
        interval: float = 0.05,  # Min time between updates
    ):
        if items is not None:
            if total is None:
                try:
                    total = len(items)  # type: ignore
                except (TypeError, AttributeError) as e:
                    raise ValueError('Cannot determine total length of iterable, please provide `total`.') from e
            self._iterator = aiter(items) if isinstance(items, AsyncIterable) else iter(items)
        elif total is not None:
            # When iterable is None, we know T must be int.
            self._iterator = cast(Iterator[T], iter(range(total)))
        else:
            raise ValueError('Must provide either `iterable` or `total`.')

        self._debouncer = debounced(interval=interval)(lambda: self._draw())
        self._draw_task = asyncio.Task(noop())

        super().__init__(total=total, description=description, initial_metrics=initial_metrics or {})

    def _draw(self):
        self._draw_task.cancel()
        super()._draw()

    def _debounced_draw(self) -> None:
        self._draw_task = self._debouncer()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            raise

    def __aiter__(self) -> AsyncIterator[T]:
        self.count = 0
        self.start_time = time.monotonic()
        self._debounced_draw()

        return async_iterator_wrapper(self, self._iterator)


class SyncProgress(ProgressBase, Iterable[T]):
    """
    A simple, Jupyter-friendly progress bar using HTML and display updates.

    - Avoids ipywidgets for better notebook saving and compatibility.
    - Designed to be theme-agnostic (light/dark).
    - Supports suffix messages and a dictionary of metrics.
    - Can be used as a context manager or iterator source (`with`, `for`).

    This progress bar features debounced updates, but only draws on the leading edge —
    so some updates may not display until the bar is closed.
    """

    def __init__(
        self,
        items: Iterable[T] | None = None,
        total: int | None = None,
        description: str = '',
        initial_metrics: dict[str, Any] | None = None,
        interval: float = 0.05,  # Min time between updates
    ):
        if items is not None:
            if total is None:
                try:
                    total = len(items)  # type: ignore
                except (TypeError, AttributeError) as e:
                    raise ValueError('Cannot determine total length of iterable, please provide `total`.') from e
            self._iterator = iter(items)
        elif total is not None:
            # When iterable is None, we know T must be int.
            self._iterator = cast(Iterator[T], iter(range(total)))
        else:
            raise ValueError('Must provide either `iterable` or `total`.')

        self._last_draw_time = 0
        self._interval = interval

        super().__init__(total=total, description=description, initial_metrics=initial_metrics or {})

    def _draw(self):
        now = time.monotonic()
        super()._draw()
        self._last_draw_time = now

    def _debounced_draw(self):
        now = time.monotonic()
        if now - self._last_draw_time >= self._interval:
            self._draw()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            raise

    def __iter__(self) -> Iterator[T]:
        self.count = 0
        self.start_time = time.monotonic()
        self._debounced_draw()

        return sync_iterator_wrapper(self, self._iterator)


async def noop():
    pass


class _DisplayInNotebook:
    def __init__(self):
        self._show = displayer()

    def show(self, ob):
        self._show(ob)

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def clear_for_print(self):
        pass

    def finalize(self):
        pass


class _DisplayInConsole:
    def __init__(self):
        self._line_length = 0

    def show(self, ob):
        # \033[K clears the line
        line = repr(ob)
        print(f'\r\033[K{line}', end='', flush=True)

    def print(self, *args, **kwargs):
        # Embed first arg in clear string to avoid unwanted separator at the start
        start = args[0] if args else ''
        if 'flush' not in kwargs:
            kwargs['flush'] = True
        print(f'\r\033[K{start}', *args[1:], **kwargs)

    def clear_for_print(self):
        print('\r\033[K', end='', flush=True)

    def finalize(self):
        print('\n', end='', flush=True)
