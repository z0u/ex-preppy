import asyncio
from typing import AsyncIterable, AsyncIterator, Generic, Iterable, Iterator, TypeVar

from utils.progress._progress import _Progress

T = TypeVar('T')


class IteratorWrapper(Generic[T], Iterator[T]):
    def __init__(self, pbar: _Progress, iterator: Iterator[T] | AsyncIterator[T]):
        self.pbar = pbar
        self.iterator = iterator

    def __iter__(self) -> Iterator[T]:
        return self

    async def __next__(self) -> T:
        try:
            if isinstance(self.iterator, AsyncIterator):
                value = await anext(self.iterator)
            else:
                try:
                    value = next(self.iterator)
                except StopIteration as e:
                    raise StopAsyncIteration from e
            self.pbar.count += 1
            return value
        except Exception:
            self.pbar.close()
            raise


def sync_iterator_wrapper(pbar: _Progress, iterator: Iterable[T]):
    try:
        for x in iterator:
            pbar.count += 1
            yield x
    finally:
        pbar.close()


async def async_iterator_wrapper(pbar: _Progress, iterator: Iterable[T] | AsyncIterable[T]):
    if not isinstance(iterator, AsyncIterable):
        iterator = _as_async(iterator)
    try:
        async for x in iterator:
            pbar.count += 1
            yield x
    finally:
        pbar.close()


class AsyncIteratorWrapper(Generic[T], AsyncIterator[T]):
    def __init__(self, pbar: _Progress, iterator: Iterator[T] | AsyncIterator[T]):
        self.pbar = pbar
        self.iterator = iterator

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        try:
            if isinstance(self.iterator, AsyncIterator):
                value = await anext(self.iterator)
            else:
                try:
                    value = next(self.iterator)
                except StopIteration as e:
                    raise StopAsyncIteration from e
            self.pbar.count += 1
            return value
        except Exception:
            self.pbar.close()
            raise


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
