import asyncio
from typing import AsyncIterable, Iterable, TypeVar

from utils.progress._progress import _Progress

T = TypeVar('T')


def sync_iterator_wrapper(pbar: _Progress, iterator: Iterable[T]):
    """Yields items and closes the bar at the end."""
    try:
        for x in iterator:
            pbar.count += 1
            yield x
    finally:
        pbar.close()


async def async_iterator_wrapper(pbar: _Progress, iterator: Iterable[T] | AsyncIterable[T]):
    """Yields items and closes the bar at the end."""
    if not isinstance(iterator, AsyncIterable):
        iterator = _as_async(iterator)
    try:
        async for x in iterator:
            pbar.count += 1
            yield x
    finally:
        pbar.close()


async def co_op(iterable: Iterable[T] | AsyncIterable[T]):
    """
    Cooperate with other async tasks by briefly sleeping when yielding elemends.

    This gives other tasks a chance to run once per iteration.
    """
    if not isinstance(iterable, AsyncIterable):
        iterable = _as_async(iterable)

    try:
        async for x in iterable:
            try:
                yield x
            finally:
                await asyncio.sleep(0)
    finally:
        # One more for StopIteration
        await asyncio.sleep(0)


async def _as_async(iterable: Iterable[T]) -> AsyncIterable[T]:
    for x in iterable:
        yield x
