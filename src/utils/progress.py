import time
from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, Generic, Iterable, Iterator, Optional, TypeVar, cast

from IPython.display import HTML

from utils.nb import displayer

T = TypeVar('T')


class _IteratorWrapper(Generic[T], Iterator[T]):
    def __init__(self, pbar: 'Progress[T]', iterator: Iterator[T]):
        self.pbar = pbar
        self.iterator = iterator

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        if self.pbar._closed:
            raise StopIteration
        try:
            value = next(self.iterator)
            self.pbar.update(1)
            return value
        except StopIteration:
            self.pbar.close()
            raise
        except Exception:
            self.pbar.close()
            raise


class Progress(Generic[T]):
    """
    A simple, Jupyter-friendly progress bar using HTML and display updates.

    Avoids ipywidgets for better notebook saving and compatibility.
    Designed to be theme-agnostic (light/dark).
    Supports suffix messages and a dictionary of metrics.
    Can be used as a context manager or an iterator source.
    """

    _iterator: Iterator[T]

    def __init__(
        self,
        iterable: Optional[Iterable[T]] = None,
        total: Optional[int] = None,
        description: str = '',
        initial_metrics: Optional[Dict[str, Any]] = None,
        min_interval_sec: float = 0.1,  # Min time between updates
    ):
        if iterable is not None:
            if total is None:
                try:
                    total = len(iterable)  # type: ignore
                except (TypeError, AttributeError) as e:
                    raise ValueError('Cannot determine total length of iterable, please provide `total`.') from e
            self._iterator = iter(iterable)
        elif total is not None:
            # When iterable is None, we know T must be int.
            self._iterator = cast(Iterator[T], iter(range(total)))
        else:
            raise ValueError('Must provide either `iterable` or `total`.')

        if total <= 0:
            raise ValueError('total must be positive')

        self.total = total
        self.description = description
        self.metrics = initial_metrics.copy() if initial_metrics else {}
        self.suffix = ''
        self.min_interval_sec = min_interval_sec

        self.count = 0
        self.start_time = time.monotonic()
        self.last_update_time = 0.0
        self._update_display = displayer()
        self._closed = False
        self._display(force=True)

    def update(
        self,
        n: int = 1,
        suffix: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Increment progress and update the display."""
        if self._closed:
            return
        self.count = min(self.count + n, self.total)
        if suffix is not None:
            self.suffix = suffix
        if metrics is not None:
            self.metrics = metrics.copy()

        self._display()

    def _display(self, force: bool = False) -> None:
        """Render and update the HTML display, respecting rate limits."""
        if self._closed:
            return

        now = time.monotonic()
        if not force:
            dt = now - self.last_update_time
            if dt < self.min_interval_sec:
                return
            if self.count >= self.total:
                return

        html_content = self._render_html()
        self._update_display(HTML(html_content))

        self.last_update_time = now

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

    def close(self) -> None:
        """Finalize the progress bar, ensuring it shows 100%."""
        if not self._closed:
            self._display(force=True)
            self._closed = True

    def __enter__(self):
        self._display(force=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self) -> Iterator[T]:
        self.count = 0
        self.start_time = time.monotonic()
        self.last_update_time = 0.0
        self._closed = False
        self._display(force=True)

        return _IteratorWrapper(self, self._iterator)


@dataclass(slots=True)
class BarData:
    count: int
    total: int
    description: str
    suffix: str
    elapsed_time: float

    @property
    def fraction(self) -> float:
        return (self.count / self.total) if self.total > 0 else 1


def format_time(seconds: float) -> str:
    if not isinstance(seconds, (int, float)) or not isfinite(seconds) or seconds < 0:
        return '??:??:??'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h == 0:
        return f'{m:02d}:{s:02d}'
    else:
        return f'{h:d}:{m:02d}:{s:02d}'


def format_progress_bar(data: BarData, metrics: dict[str, Any]):
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

    bar_text_elem = f'<b>{data.description}</b>: {data.fraction * 100:.1f}% [<b>{data.count}</b>/{data.total}]'
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


def format_metrics(metrics: dict[str, Any]):
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
