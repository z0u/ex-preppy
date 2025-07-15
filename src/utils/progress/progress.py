import asyncio
import time
from math import isfinite
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    TypeVar,
    cast,
    override,
)

from IPython.display import HTML

from utils.coro import debounced
from utils.nb import displayer
from utils.progress._progress import _Progress
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
        interval: float = 0.1,  # Min time between updates
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
        return format_progress_bar(data, self.metrics)

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


def format_time(seconds: float) -> str:
    if not isinstance(seconds, (int, float)) or not isfinite(seconds) or seconds < 0:
        return '??:??:??'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h == 0:
        return f'{m:02d}:{s:02d}'
    else:
        return f'{h:d}:{m:02d}:{s:02d}'


def format_progress_bar(data: BarData, metrics: Mapping[str, Any]):
    return f"""
    <div style="width: 100%; padding: 5px 0; font-family: monospace;">
        {format_bar(data, format_bar_text(data))}
        {format_metrics(metrics)}
    </div>
    """


def format_bar_text(data: BarData):
    items_per_sec = data.count / data.elapsed_time if data.elapsed_time > 0 else 0
    eta_sec = (data.total - data.count) / items_per_sec if items_per_sec > 0 and data.count < data.total else 0
    elapsed_str = format_time(data.elapsed_time)
    eta_str = format_time(eta_sec) if data.count < data.total else format_time(0)

    bar_text_elem = ''
    if data.description:
        bar_text_elem += f'<b>{data.description}</b>: '
    bar_text_elem += f'{data.fraction * 100:.1f}% [<b>{data.count}</b>/{data.total}]'
    if data.suffix:
        bar_text_elem += f' | <b>{data.suffix}</b>'
    bar_text_elem += f' [<b>{elapsed_str}</b>/<{eta_str}, {items_per_sec:.2f} it/s]'
    return bar_text_elem


def format_bar(data: BarData, bar_text_elem: str) -> str:
    html = f"""
        <!-- Progress bar container -->
        <div style="position: relative; height: calc(1em * 5/3); width: 100%;">
            <!-- Triangle indicator -->
            <div style="position: absolute; bottom: -4px; left: calc({data.fraction * 100:.1f}% - 4px);">
                <div style="
                    width: 0;
                    height: 0;
                    border-left: 4px solid transparent;
                    border-right: 4px solid transparent;
                    border-bottom: 4px solid currentColor;
                "></div>
            </div>
            <!-- Progress bar -->
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                width: {data.fraction * 100:.1f}%;
                background-color: color(from currentColor srgb r g b / 0.1);
                border-bottom: 1px solid currentColor;
            "></div>
            <!-- Text overlay -->
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                text-align: center;
                line-height: calc(1em * 5/3);
                font-size: 0.9em;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                border-bottom: 1px dashed color(from currentColor srgb r g b / 0.5);
            ">
                {bar_text_elem}
            </div>
        </div>
    """

    return html


def format_metrics(metrics: Mapping[str, Any]):
    html = f"""
    <div style="
        display: grid;
        grid-template-columns: repeat({len(metrics)}, minmax(80px, 1fr));
        gap: 5px 0px;
        width: 100%;
        margin-top: 10px;
        font-size: 0.85em;
    ">"""

    for key in metrics.keys():
        html += f"""<div style="
            font-weight: bold;
            border-bottom: 1px solid currentColor;
            padding-block: 2px;
            padding-inline: 10px;
            text-align: left;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        ">{key}</div>"""

    for value in metrics.values():
        if isinstance(value, float):
            val_str = f'{value:.4g}'
        else:
            val_str = str(value)
        html += f"""<div style="
            padding-block: 2px;
            padding-inline: 10px;
            text-align: left;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        ">{val_str}</div>"""

    html += '</div>'

    return html


async def noop():
    pass


# Example Usage (for testing in a notebook cell):
# from src.utils.progress import Progress
# import time
#
# total_items = 150
# items = range(total_items)
# metrics = {"loss": 1.5, "accuracy": 0.3}
#
# print("Iterating over Progress(range(total_items)):")
# with Progress(items, description="Training Epoch 1", initial_metrics=metrics) as pbar:
#     for i in pbar:
#         time.sleep(0.05)
#         if i % 10 == 0 and i > 0:
#             new_metrics = pbar.metrics.copy()
#             new_metrics["loss"] -= 0.05
#             new_metrics["accuracy"] += 0.02
#             pbar.update(0, metrics=new_metrics)
#         if i == 100:
#             pbar.update(0, suffix="Halfway there!")
#
# print("\nIterating over Progress(total=total_items):")
# with Progress(total=total_items, description="Processing Items") as pbar:
#     for i in pbar:
#         time.sleep(0.02)
#         if i == 50:
#             pbar.update(0, metrics={"status": "Phase 1 Complete"})
