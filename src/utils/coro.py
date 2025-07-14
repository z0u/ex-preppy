import asyncio
import time
from functools import wraps
from inspect import isawaitable
from typing import Any, Callable, ParamSpec, Union, overload

P = ParamSpec('P')


@overload
def debounced(func: Callable[P, Any]) -> Callable[P, Any]: ...


@overload
def debounced(*, min_interval: float = 0.0) -> Callable[[Callable[P, Any]], Callable[P, Any]]: ...


def debounced(  # noqa: C901
    func: Callable[P, Any] | None = None, *, min_interval: float = 0.0
) -> Union[Callable[P, Any], Callable[[Callable[P, Any]], Callable[P, Any]]]:
    """
    Debounce decorator with leading and trailing edge execution.

    Works in both sync and async contexts:
    - With event loop: Uses asyncio.Task for proper async debouncing
    - Without event loop: Executes immediately (no debouncing, falls back to caller)

    - Leading edge: Function runs immediately on first call
    - Trailing edge: If called during execution, runs again with latest arguments
    - Latest arguments: Always uses the most recent arguments for trailing execution
    - min_interval: Minimum time (in seconds) between function executions

    Args:
        func: Function to debounce (None when used with parameters)
        min_interval: Minimum time in seconds between executions (default: 0.0)

    Returns:
        Debounced function that works in both sync and async contexts
    """

    def decorator(f: Callable[P, Any]) -> Callable[P, Any]:  # noqa: C901
        # Async state
        pending: tuple[tuple, dict] | None = None
        current_task: asyncio.Task | None = None
        async_last_execution_time: float = 0.0

        async def _execute_loop():
            nonlocal pending, current_task, async_last_execution_time
            try:
                while pending is not None:
                    current_args, current_kwargs = pending
                    pending = None

                    # Respect minimum interval
                    if min_interval > 0.0:
                        now = time.monotonic()
                        time_since_last = now - async_last_execution_time
                        if time_since_last < min_interval:
                            await asyncio.sleep(min_interval - time_since_last)

                    # Execute function
                    ret = f(*current_args, **current_kwargs)
                    if isawaitable(ret):
                        await ret

                    async_last_execution_time = time.monotonic()
            finally:
                current_task = None

        def _handle_async_call(*args: P.args, **kwargs: P.kwargs) -> asyncio.Task:
            """Handle debounced function call in async context."""
            nonlocal pending, current_task
            # Store the latest arguments
            pending = (args, kwargs)

            # If not running, start execution (leading edge)
            if not current_task or current_task.done():
                current_task = asyncio.create_task(_execute_loop())

            # Return current task (caller can await if desired)
            return current_task

        def _handle_sync_call(*args: P.args, **kwargs: P.kwargs) -> Any:
            """Handle debounced function call in sync context."""
            # No event loop, execute immediately and let caller handle debouncing
            ret = f(*args, **kwargs)
            if isawaitable(ret):
                # This shouldn't happen in sync context, but handle gracefully
                pass
            return ret

        @wraps(f)
        def _debounced(*args: P.args, **kwargs: P.kwargs) -> Any:
            """
            Call the debounced function.

            Returns:
                - asyncio.Task in async context (can be awaited)
                - None in sync context (caller handles debouncing)
            """
            # Check if we're in an async context
            try:
                asyncio.get_running_loop()
                return _handle_async_call(*args, **kwargs)
            except RuntimeError:
                return _handle_sync_call(*args, **kwargs)

        return _debounced

    # Support both @debounced and @debounced(min_interval=0.1)
    if func is None:
        return decorator
    else:
        return decorator(func)
