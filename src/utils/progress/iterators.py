import asyncio
from typing import AsyncIterator, Generic, Iterator, TypeVar

from utils.progress._progress import _Progress

T = TypeVar('T')


class AsyncIteratorWrapper(Generic[T], AsyncIterator[T]):
    def __init__(self, pbar: _Progress, iterator: Iterator[T] | AsyncIterator[T], auto_yield: bool):
        self.pbar = pbar
        self.iterator = iterator
        self.auto_yield = auto_yield

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
        finally:
            if self.auto_yield:
                await asyncio.sleep(0)
