import inspect
from functools import wraps
from typing import Any, Coroutine, Callable, ParamSpec, TypeVar, cast

P = ParamSpec('P')
R = TypeVar('R')


def coerce_to_async(fn: Callable[P, R | Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
    if inspect.iscoroutinefunction(fn):
        return cast(Callable[P, Coroutine[Any, Any, R]], fn)

    fn = cast(Callable[P, R], fn)

    @wraps(fn)
    async def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper
