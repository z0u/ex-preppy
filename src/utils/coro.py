import asyncio
import time
from functools import wraps
from inspect import isawaitable
from typing import Any, Callable, ParamSpec, Union, overload

P = ParamSpec('P')


@overload
def debounced(func: Callable[P, Any]) -> Callable[P, asyncio.Task]: ...


@overload
def debounced(*, min_interval: float = 0.0) -> Callable[[Callable[P, Any]], Callable[P, asyncio.Task]]: ...


def debounced(
    func: Callable[P, Any] | None = None, *, min_interval: float = 0.0
) -> Union[Callable[P, asyncio.Task], Callable[[Callable[P, Any]], Callable[P, asyncio.Task]]]:
    """
    Debounce decorator with leading and trailing edge execution.

    - Leading edge: Function runs immediately on first call
    - Trailing edge: If called during execution, runs again with latest arguments
    - Latest arguments: Always uses the most recent arguments for trailing execution
    - min_interval: Minimum time (in seconds) between function executions

    Args:
        func: Function to debounce (None when used with parameters)
        min_interval: Minimum time in seconds between executions (default: 0.0)

    Returns:
        Debounced function that returns an asyncio.Task
    """

    def decorator(f: Callable[P, Any]) -> Callable[P, asyncio.Task]:
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
                    if min_interval > 0.0:
                        now = time.monotonic()
                        time_since_last = now - last_execution_time
                        if time_since_last < min_interval:
                            await asyncio.sleep(min_interval - time_since_last)

                    # Execute function
                    ret = f(*current_args, **current_kwargs)
                    if isawaitable(ret):
                        await ret

                    last_execution_time = time.monotonic()
            finally:
                current_task = None

        @wraps(f)
        def _debounced(*args: P.args, **kwargs: P.kwargs) -> asyncio.Task:
            """
            Call the debounced function.

            Returns:
                - asyncio.Task if starting new execution (can be awaited)
                - Current task if execution is already running
            """
            nonlocal pending, current_task
            # Store the latest arguments
            pending = (args, kwargs)

            # If not running, start execution (leading edge)
            if not current_task or current_task.done():
                current_task = asyncio.create_task(_execute_loop())

            # Return current task (caller can await if desired)
            return current_task

        return _debounced

    # Support both @debounced and @debounced(min_interval=0.1)
    if func is None:
        return decorator
    else:
        return decorator(func)
