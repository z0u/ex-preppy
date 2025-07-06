import asyncio
from functools import wraps
from inspect import isawaitable
from typing import Any, Callable, ParamSpec

P = ParamSpec('P')


def debounced(func: Callable[P, Any]) -> Callable[P, asyncio.Task]:
    """
    Debounce decorator with leading and trailing edge execution.

    - Leading edge: Function runs immediately on first call
    - Trailing edge: If called during execution, runs again with latest arguments
    - Latest arguments: Always uses the most recent arguments for trailing execution
    """
    pending: tuple[tuple, dict] | None = None
    current_task: asyncio.Task | None = None

    async def _execute_loop():
        nonlocal pending, current_task
        try:
            while pending is not None:
                current_args, current_kwargs = pending
                pending = None
                ret = func(*current_args, **current_kwargs)
                if isawaitable(ret):
                    await ret
        finally:
            current_task = None

    @wraps(func)
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
