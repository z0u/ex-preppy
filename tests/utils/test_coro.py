import asyncio
import time
from unittest.mock import Mock

import pytest

from utils.coro import debounced


class TestDebounced:
    """Test the debounced decorator functionality."""

    @pytest.mark.asyncio
    async def test_basic_debounced_functionality(self):
        """Test basic debouncing without timing constraints."""
        mock_func = Mock()

        @debounced
        def test_func(arg):
            mock_func(arg)

        # Multiple rapid calls should result in leading + trailing execution
        task1 = test_func("first")  # type: ignore
        task2 = test_func("second")  # type: ignore
        task3 = test_func("third")  # type: ignore

        # All tasks should be the same (since debounced)
        assert task1 is task2 is task3

        # Wait for execution
        await task1  # type: ignore

        # Function should be called with latest arguments
        assert mock_func.call_count >= 1
        # Latest call should have "third" as argument
        last_call_args = mock_func.call_args_list[-1]
        assert last_call_args[0][0] == "third"

    @pytest.mark.asyncio
    async def test_debounced_with_timing(self):
        """Test debounced decorator with minimum interval."""
        calls = []

        @debounced(min_interval=0.1)
        def test_func(arg):
            calls.append((arg, time.monotonic()))

        start_time = time.monotonic()

        # First call should execute immediately (leading edge)
        task1 = test_func("first")  # type: ignore

        # Rapid subsequent calls should be debounced
        await asyncio.sleep(0.01)  # Small delay
        task2 = test_func("second")  # type: ignore
        task3 = test_func("third")  # type: ignore

        # Wait for completion
        await task1
        await task2
        await task3

        end_time = time.monotonic()

        # Should have exactly 2 calls: leading + trailing
        assert len(calls) == 2
        assert calls[0][0] == "first"  # Leading edge
        assert calls[1][0] == "third"  # Trailing edge with latest args

        # Check timing - should take at least min_interval
        total_time = end_time - start_time
        assert total_time >= 0.1

        # Check that there was proper spacing between calls
        call_gap = calls[1][1] - calls[0][1]
        assert call_gap >= 0.1

    @pytest.mark.asyncio
    async def test_debounced_backwards_compatibility(self):
        """Test that decorator without parameters works as before."""
        calls = []

        @debounced  # No parameters
        def test_func(arg):
            calls.append(arg)

        # Multiple calls
        task1 = test_func("first")  # type: ignore
        task2 = test_func("second")  # type: ignore

        await task1  # type: ignore
        await task2  # type: ignore

        # Should execute with latest args (basic debouncing)
        assert "second" in calls

    @pytest.mark.asyncio
    async def test_debounced_with_async_function(self):
        """Test debounced decorator with async function."""
        calls = []

        @debounced(min_interval=0.05)
        async def async_test_func(arg):
            calls.append(arg)
            await asyncio.sleep(0.01)  # Simulate async work

        # Multiple calls
        task1 = async_test_func("first")  # type: ignore
        task2 = async_test_func("second")  # type: ignore

        await task1
        await task2

        # Should work with async functions too
        assert len(calls) >= 1
        assert "second" in calls

    def test_debounced_parameter_syntax(self):
        """Test that decorator can be used with parameters."""
        # This should not raise an error
        @debounced(min_interval=0.1)
        def test_func():
            pass

        # And this should also work (backwards compatibility)
        @debounced
        def test_func2():
            pass

        assert callable(test_func)
        assert callable(test_func2)
