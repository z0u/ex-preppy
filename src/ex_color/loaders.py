import logging
from typing import Iterable, Iterator

log = logging.getLogger(__name__)

def reiterate[T](it: Iterable[T]) -> Iterator[T]:
    """
    Iterates over an iterable indefinitely.

    When the iterable is exhausted, it starts over from the beginning. Unlike
    `itertools.cycle`, yielded values are not cached — so each iteration may be
    different.
    """
    while True:
        yield from it
