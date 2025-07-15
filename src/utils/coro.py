import asyncio
from functools import wraps
from inspect import isawaitable
import time
from typing import Any, Awaitable, Callable, ParamSpec, overload

P = ParamSpec('P')


@overload
def debounced(func: Callable[P, Any]) -> Callable[P, asyncio.Task[None]]: ...


@overload
def debounced(*, interval: float = 0.0) -> Callable[[Callable[P, Any]], Callable[P, asyncio.Task[None]]]: ...


def debounced(
    func: Callable[P, Any] | None = None,
    *,
    interval: float = 0.0,
) -> Callable[P, asyncio.Task[None]] | Callable[[Callable[P, Any]], Callable[P, asyncio.Task[None]]]:
    """
    Debounce decorator with leading and trailing edge execution.

    - Leading edge: Function runs immediately on first call
    - Trailing edge: If called during execution, runs again with latest arguments
    - Latest arguments: Always uses the most recent arguments for trailing execution
    """

    def decorator(fn: Callable[P, Any | Awaitable[Any]]):
        pending: tuple[tuple, dict] | None = None
        current_task: asyncio.Task | None = None
        last_execution_time: float = 0.0

        async def _execute_loop():
            nonlocal pending, current_task, last_execution_time
            try:
                while pending is not None:
                    current_args, current_kwargs = pending
                    pending = None

                    # Respect minimum interval
                    if interval > 0.0:
                        now = time.monotonic()
                        time_since_last = now - last_execution_time
                        if time_since_last < interval:
                            await asyncio.sleep(interval - time_since_last)

                    # Execute function
                    ret = fn(*current_args, **current_kwargs)
                    if isawaitable(ret):
                        await ret

                    last_execution_time = time.monotonic()
            finally:
                current_task = None

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> asyncio.Task:
            """
            Call the debounced function.

            Returns:
                - asyncio.Task if starting new execution (can be awaited)
                - Current task if execution is already running
            """
            nonlocal pending, current_task
            # Store the latest arguments
            pending = (args, kwargs)

            # If not running, start execution
            if not current_task or current_task.done():
                current_task = asyncio.create_task(_execute_loop())

            # Return current task (caller can await if desired)
            return current_task

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)
