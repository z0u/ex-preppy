"""Modal-based distributed progress bar components."""

import asyncio
import time
from typing import Any, Generic, TypeVar, override

import modal

from utils.nb import displayer, is_graphical_notebook
from utils.progress._progress import _Progress
from utils.progress.html import render_progress_bar
from utils.progress.model import BarData, Mark

T = TypeVar('T')


class ModalProgressWriter(_Progress, Generic[T]):
    """Remote writer that updates progress to Modal Dict with debounced behavior."""

    def __init__(
        self,
        total: int,
        description: str,
        initial_metrics: dict[str, Any] | None = None,
        dict_name: str = 'progress-bar',
        interval: float = 0.3,
        shard_size: int = 100,
    ):
        if total < 0:
            raise ValueError('total must be non-negative')

        self._dict_name = dict_name
        self._modal_dict: modal.Dict | None = None
        self._interval = interval
        self._shard_size = shard_size
        self._current_shard = 0
        self._markers_in_current_shard = 0
        self._meta_version = 1
        self._metrics_version = 1
        self._bar_version = 1

        # Initialize Modal Dict
        try:
            self._modal_dict = modal.Dict.from_name(dict_name, create_if_missing=True)
            # Initialize metadata
            self._update_metadata()
        except Exception:
            # Fall back to no-op if Modal isn't available
            pass

        # Setting these usually triggers a draw; see _Progress
        self._draw_on_change = False
        self.total = total
        self.description = description
        self.metrics = initial_metrics or {}
        self.markers = []
        self.suffix = ''
        self.start_time = time.monotonic()
        self.count = 0
        self._last_update_time = 0
        self._draw_on_change = True

    def _update_metadata(self):
        """Update metadata in Modal Dict."""
        if not self._modal_dict:
            return

        meta = {
            'version': self._meta_version,
            'bar_version': self._bar_version,
            'metrics_version': self._metrics_version,
            'max_shard': self._current_shard,
            'last_updated': time.monotonic(),
        }
        self._modal_dict['meta'] = meta
        self._meta_version += 1

    def _update_bar_data(self):
        """Update bar data in Modal Dict."""
        if not self._modal_dict:
            return

        data = BarData(
            count=self.count,
            total=self.total,
            description=self.description,
            suffix=self.suffix,
            elapsed_time=time.monotonic() - self.start_time,
        )
        self._modal_dict['bar'] = data
        self._bar_version += 1

    def _update_metrics(self):
        """Update metrics in Modal Dict."""
        if not self._modal_dict:
            return

        self._modal_dict['metrics'] = dict(self.metrics)
        self._metrics_version += 1

    def _update_markers_shard(self):
        """Update current markers shard in Modal Dict."""
        if not self._modal_dict:
            return

        # Get markers for current shard
        start_idx = self._current_shard * self._shard_size
        end_idx = start_idx + self._shard_size
        shard_markers = self.markers[start_idx:end_idx]

        if shard_markers:
            self._modal_dict[f'markers-{self._current_shard}'] = shard_markers

    def mark(self, label: str, count: int | None = None):
        """Add a marker and update shard if necessary."""
        self.markers.append(Mark(count if count is not None else self.count - 1, label))

        # Check if we need to move to a new shard
        if len(self.markers) > (self._current_shard + 1) * self._shard_size:
            self._update_markers_shard()
            self._current_shard += 1

    @override
    def _on_change(self):
        if self._draw_on_change:
            self._debounced_update()

    def _debounced_update(self):
        """Update Modal Dict with debounced behavior."""
        now = time.monotonic()
        if now - self._last_update_time >= self._interval:
            self._update_modal_dict()
            self._last_update_time = now

    def _update_modal_dict(self):
        """Update all data in Modal Dict."""
        if not self._modal_dict:
            return

        self._update_bar_data()
        self._update_metrics()
        self._update_markers_shard()
        self._update_metadata()

    @override
    def close(self) -> None:
        """Final update to Modal Dict."""
        self._update_modal_dict()

    def print(self, *args, **kwargs):
        """Print to stdout (no-op in distributed mode)."""
        # In distributed mode, printing is handled by the reader
        pass

    def update(
        self,
        total: int | None = None,
        count: int | None = None,
        description: str | None = None,
        metrics: dict[str, Any] | None = None,
        suffix: str | None = None,
    ):
        """Update the progress bar with new values."""
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


class ModalProgressReader:
    """Local reader that polls Modal Dict and displays progress."""

    def __init__(
        self,
        dict_name: str = 'progress-bar',
        poll_interval: float = 0.1,
    ):
        self._dict_name = dict_name
        self._poll_interval = poll_interval
        self._modal_dict: modal.Dict | None = None
        self._last_meta_version = 0
        self._last_bar_version = 0
        self._last_metrics_version = 0
        self._cached_bar_data: BarData | None = None
        self._cached_metrics: dict[str, Any] = {}
        self._cached_markers: list[Mark] = []
        self._running = False

        if is_graphical_notebook():
            self._show = _DisplayInNotebook()
        else:
            self._show = _DisplayInConsole()

        # Initialize Modal Dict
        try:
            self._modal_dict = modal.Dict.from_name(dict_name, create_if_missing=True)
        except Exception:
            # Fall back to no-op if Modal isn't available
            pass

    async def start_polling(self):
        """Start polling the Modal Dict for updates."""
        self._running = True
        while self._running:
            await self._poll_and_update()
            await asyncio.sleep(self._poll_interval)

    def stop_polling(self):
        """Stop polling."""
        self._running = False

    async def _poll_and_update(self):
        """Poll Modal Dict and update display if needed."""
        if not self._modal_dict:
            return

        try:
            # Check metadata first
            meta = self._modal_dict.get('meta', {})
            if not meta or meta.get('version', 0) <= self._last_meta_version:
                return

            self._last_meta_version = meta['version']

            # Update bar data if changed
            if meta.get('bar_version', 0) > self._last_bar_version:
                self._cached_bar_data = self._modal_dict.get('bar')
                self._last_bar_version = meta['bar_version']

            # Update metrics if changed
            if meta.get('metrics_version', 0) > self._last_metrics_version:
                self._cached_metrics = self._modal_dict.get('metrics', {})
                self._last_metrics_version = meta['metrics_version']

            # Load all marker shards
            max_shard = meta.get('max_shard', 0)
            all_markers = []
            for shard_idx in range(max_shard + 1):
                shard_markers = self._modal_dict.get(f'markers-{shard_idx}', [])
                all_markers.extend(shard_markers)
            self._cached_markers = all_markers

            # Update display
            self._update_display()

        except Exception:
            # Ignore errors during polling
            pass

    def _update_display(self):
        """Update the progress bar display."""
        if not self._cached_bar_data:
            return

        # Create a display object with HTML representation
        display_obj = _ModalProgressDisplay(
            self._cached_bar_data,
            self._cached_metrics,
            self._cached_markers,
        )
        self._show.show(display_obj)

    def print(self, *args, **kwargs):
        """Print without damaging the progress bar display."""
        self._show.print(*args, **kwargs)

    def finalize(self):
        """Finalize the display."""
        self._show.finalize()


class _ModalProgressDisplay:
    """Display object for Modal progress that mimics the existing progress interface."""

    def __init__(self, bar_data: BarData, metrics: dict[str, Any], markers: list[Mark]):
        self.bar_data = bar_data
        self.metrics = metrics
        self.markers = markers

    def __repr__(self) -> str:
        return f'{self.bar_data.description}: {self.bar_data.fraction:.1%} [{self.bar_data.count:d}/{self.bar_data.total:d}]'

    def _repr_html_(self) -> str:
        """Generate the HTML for the progress bar and metrics grid."""
        return render_progress_bar(self.bar_data, self.metrics, self.markers)


class _DisplayInNotebook:
    def __init__(self):
        self._show = displayer()

    def show(self, ob):
        self._show(ob)

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

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

    def finalize(self):
        print('\n', end='', flush=True)
