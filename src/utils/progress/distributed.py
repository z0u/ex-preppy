"""Distributed progress bar classes that automatically choose between local and Modal modes."""

import asyncio
import time
from typing import Any, AsyncIterable, AsyncIterator, Generic, Iterable, Iterator, TypeVar, cast

import modal

from utils.progress.modal_progress import ModalProgressWriter
from utils.progress.progress import Progress, SyncProgress

T = TypeVar('T')


def is_modal_environment() -> bool:
    """Check if we're running in a Modal environment."""
    try:
        return not modal.is_local()
    except Exception:
        return False


class DistributedProgress(Generic[T]):
    """
    Progress bar that automatically chooses between local and distributed modes.

    In local environments, uses the standard Progress class.
    In Modal environments, uses ModalProgressWriter to send updates to a shared dict.
    """

    def __init__(
        self,
        items: Iterable[T] | AsyncIterable[T] | None = None,
        total: int | None = None,
        description: str = '',
        initial_metrics: dict[str, Any] | None = None,
        interval: float = 0.05,
        dict_name: str = 'progress-bar',
    ):
        self._dict_name = dict_name

        if is_modal_environment():
            # Use Modal writer in distributed environment
            if items is not None:
                if total is None:
                    try:
                        total = len(items)  # type: ignore
                    except (TypeError, AttributeError) as e:
                        raise ValueError('Cannot determine total length of iterable, please provide `total`.') from e
                self._iterator = aiter(items) if isinstance(items, AsyncIterable) else iter(items)
            elif total is not None:
                self._iterator = cast(Iterator[T], iter(range(total)))
            else:
                raise ValueError('Must provide either `iterable` or `total`.')

            self._progress = ModalProgressWriter(
                total=total,
                description=description,
                initial_metrics=initial_metrics or {},
                dict_name=dict_name,
                interval=interval,
            )
        else:
            # Use standard progress in local environment
            self._progress = Progress(
                items=items,
                total=total,
                description=description,
                initial_metrics=initial_metrics,
                interval=interval,
            )
            self._iterator = None

    @property
    def total(self):
        return self._progress.total

    @total.setter
    def total(self, value):
        self._progress.total = value

    @property
    def count(self):
        return self._progress.count

    @count.setter
    def count(self, value):
        self._progress.count = value

    @property
    def description(self):
        return self._progress.description

    @description.setter
    def description(self, value):
        self._progress.description = value

    @property
    def suffix(self):
        return self._progress.suffix

    @suffix.setter
    def suffix(self, value):
        self._progress.suffix = value

    @property
    def metrics(self):
        return self._progress.metrics

    @metrics.setter
    def metrics(self, value):
        self._progress.metrics = value

    def update(self, **kwargs):
        """Update the progress bar."""
        if hasattr(self._progress, 'update'):
            self._progress.update(**kwargs)

    def mark(self, label: str, count: int | None = None):
        """Add a marker."""
        self._progress.mark(label, count)

    def print(self, *args, **kwargs):
        """Print without damaging the progress bar display."""
        self._progress.print(*args, **kwargs)

    def close(self):
        """Close the progress bar."""
        self._progress.close()

    async def __aenter__(self):
        if hasattr(self._progress, '__aenter__'):
            return await self._progress.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._progress, '__aexit__'):
            await self._progress.__aexit__(exc_type, exc_val, exc_tb)
        else:
            self.close()

    def __enter__(self):
        if hasattr(self._progress, '__enter__'):
            return self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._progress, '__exit__'):
            return self._progress.__exit__(exc_type, exc_val, exc_tb)
        else:
            self.close()

    def __aiter__(self) -> AsyncIterator[T]:
        if hasattr(self._progress, '__aiter__'):
            return self._progress.__aiter__()
        elif self._iterator:
            return self._async_iterator_wrapper()
        else:
            raise TypeError('Object is not async iterable')

    def __iter__(self) -> Iterator[T]:
        if hasattr(self._progress, '__iter__'):
            return self._progress.__iter__()
        elif self._iterator:
            return self._sync_iterator_wrapper()
        else:
            raise TypeError('Object is not iterable')

    async def _async_iterator_wrapper(self) -> AsyncIterator[T]:
        """Async iterator wrapper for Modal writer."""
        self.count = 0
        self._progress.start_time = time.monotonic()

        if isinstance(self._iterator, AsyncIterator):
            async for item in self._iterator:
                yield item
                self.count += 1
        else:
            for item in self._iterator:
                yield item
                self.count += 1
                await asyncio.sleep(0)  # Yield control

    def _sync_iterator_wrapper(self) -> Iterator[T]:
        """Sync iterator wrapper for Modal writer."""
        self.count = 0
        self._progress.start_time = time.monotonic()

        for item in self._iterator:
            yield item
            self.count += 1


class DistributedSyncProgress(Generic[T]):
    """Sync progress bar that automatically chooses between local and distributed modes."""

    def __init__(
        self,
        items: Iterable[T] | None = None,
        total: int | None = None,
        description: str = '',
        initial_metrics: dict[str, Any] | None = None,
        interval: float = 0.05,
        dict_name: str = 'progress-bar',
    ):
        self._dict_name = dict_name

        if is_modal_environment():
            # Use Modal writer in distributed environment
            if items is not None:
                if total is None:
                    try:
                        total = len(items)  # type: ignore
                    except (TypeError, AttributeError) as e:
                        raise ValueError('Cannot determine total length of iterable, please provide `total`.') from e
                self._iterator = iter(items)
            elif total is not None:
                self._iterator = cast(Iterator[T], iter(range(total)))
            else:
                raise ValueError('Must provide either `iterable` or `total`.')

            self._progress = ModalProgressWriter(
                total=total,
                description=description,
                initial_metrics=initial_metrics or {},
                dict_name=dict_name,
                interval=interval,
            )
        else:
            # Use standard sync progress in local environment
            self._progress = SyncProgress(
                items=items,
                total=total,
                description=description,
                initial_metrics=initial_metrics,
                interval=interval,
            )
            self._iterator = None

    @property
    def total(self):
        return self._progress.total

    @total.setter
    def total(self, value):
        self._progress.total = value

    @property
    def count(self):
        return self._progress.count

    @count.setter
    def count(self, value):
        self._progress.count = value

    @property
    def description(self):
        return self._progress.description

    @description.setter
    def description(self, value):
        self._progress.description = value

    @property
    def suffix(self):
        return self._progress.suffix

    @suffix.setter
    def suffix(self, value):
        self._progress.suffix = value

    @property
    def metrics(self):
        return self._progress.metrics

    @metrics.setter
    def metrics(self, value):
        self._progress.metrics = value

    def update(self, **kwargs):
        """Update the progress bar."""
        if hasattr(self._progress, 'update'):
            self._progress.update(**kwargs)

    def mark(self, label: str, count: int | None = None):
        """Add a marker."""
        self._progress.mark(label, count)

    def print(self, *args, **kwargs):
        """Print without damaging the progress bar display."""
        self._progress.print(*args, **kwargs)

    def close(self):
        """Close the progress bar."""
        self._progress.close()

    def __enter__(self):
        if hasattr(self._progress, '__enter__'):
            return self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._progress, '__exit__'):
            return self._progress.__exit__(exc_type, exc_val, exc_tb)
        else:
            self.close()

    def __iter__(self) -> Iterator[T]:
        if hasattr(self._progress, '__iter__'):
            return self._progress.__iter__()
        elif self._iterator:
            return self._sync_iterator_wrapper()
        else:
            raise TypeError('Object is not iterable')

    def _sync_iterator_wrapper(self) -> Iterator[T]:
        """Sync iterator wrapper for Modal writer."""
        self.count = 0
        self._progress.start_time = time.monotonic()

        for item in self._iterator:
            yield item
            self.count += 1
