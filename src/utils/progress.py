import time
from math import isfinite
from typing import Any, Dict, Generic, Iterable, Iterator, Optional, TypeVar, cast

from IPython.display import HTML

from utils.nb import displayer

T = TypeVar('T', default=int)


class _CounterIterator(Iterator[int]):
    def __init__(self, pbar: 'RichProgress[Any]'):
        self.pbar = pbar

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        if self.pbar._closed:
            raise StopIteration
        if self.pbar.count >= self.pbar.total:
            self.pbar.close()
            raise StopIteration

        current_count = self.pbar.count
        self.pbar.update(1)
        return current_count


class _IteratorWrapper(Generic[T], Iterator[T]):
    def __init__(self, pbar: 'RichProgress[T]', iterator: Iterator[T]):
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


class RichProgress(Generic[T]):
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
        percentage = (self.count / self.total) * 100 if self.total > 0 else 100
        elapsed_time = time.monotonic() - self.start_time
        items_per_sec = self.count / elapsed_time if elapsed_time > 0 else 0
        eta_sec = (self.total - self.count) / items_per_sec if items_per_sec > 0 and self.count < self.total else 0

        elapsed_str = format_time(elapsed_time)
        eta_str = format_time(eta_sec) if self.count < self.total else format_time(0)

        bar_text = f'<b>{self.description}</b>: {percentage:.1f}% [<b>{self.count}</b>/{self.total}]'
        if self.suffix:
            bar_text += f' | <b>{self.suffix}</b>'
        bar_text += f' [<b>{elapsed_str}</b>/<{eta_str}, {items_per_sec:.2f} it/s]'

        html = f"""
        <div style="width: 100%; padding: 5px 0; font-family: monospace;">
            <!-- Progress bar container -->
            <div style="position: relative; height: calc(1em * 5/3); width: 100%;">
                <!-- Triangle indicator -->
                <div style="position: absolute; bottom: -4px; left: calc({percentage:.1f}% - 4px);">
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
                    width: {percentage:.1f}%;
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
                    {bar_text}
                </div>
            </div>
        """

        if self.metrics:
            num_metrics = len(self.metrics)

            html += f"""
            <div style="
                display: grid;
                grid-template-columns: repeat({num_metrics}, minmax(80px, 1fr));
                gap: 5px 0px;
                width: 100%;
                margin-top: 10px;
                font-size: 0.85em;
            ">"""

            for key in self.metrics.keys():
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

            for value in self.metrics.values():
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

        html += '</div>'
        return html

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


def format_time(seconds: float) -> str:
    if not isinstance(seconds, (int, float)) or not isfinite(seconds) or seconds < 0:
        return '??:??:??'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h == 0:
        return f'{m:02d}:{s:02d}'
    else:
        return f'{h:d}:{m:02d}:{s:02d}'


# Example Usage (for testing in a notebook cell):
# from src.utils.progress import RichProgress
# import time
#
# total_items = 150
# items = range(total_items)
# metrics = {"loss": 1.5, "accuracy": 0.3}
#
# print("Iterating over RichProgress(range(total_items)):")
# with RichProgress(items, description="Training Epoch 1", initial_metrics=metrics) as pbar:
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
# print("\nIterating over RichProgress(total=total_items):")
# with RichProgress(total=total_items, description="Processing Items") as pbar:
#     for i in pbar:
#         time.sleep(0.02)
#         if i == 50:
#             pbar.update(0, metrics={"status": "Phase 1 Complete"})
