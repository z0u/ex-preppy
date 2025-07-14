import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from utils.progress import Progress


class TestProgressDebouncing:
    """Test Progress class debouncing functionality."""

    @patch('utils.progress.displayer')
    def test_progress_basic_functionality(self, mock_displayer):
        """Test that basic progress functionality still works."""
        mock_display = Mock()
        mock_displayer.return_value = mock_display

        # Create progress bar
        with Progress(total=5, description="Test", min_interval_sec=0.1) as pbar:
            for _i in range(5):
                pbar.update(1)

        # Should have made display calls
        assert mock_display.call_count >= 1

    @patch('utils.progress.displayer')
    def test_progress_rapid_updates_fallback(self, mock_displayer):
        """Test rapid updates fall back to manual rate limiting when no event loop."""
        mock_display = Mock()
        mock_displayer.return_value = mock_display

        # Create progress bar with short interval
        with Progress(total=10, description="Rapid", min_interval_sec=0.05) as pbar:
            time.monotonic()

            # Rapid updates (no sleep)
            for _i in range(10):
                pbar.update(1)

            time.monotonic()

        # Should have limited display calls due to rate limiting
        # (Exact count depends on timing, but should be less than 10)
        assert mock_display.call_count < 10
        assert mock_display.call_count >= 2  # At least initial + final

    @patch('utils.progress.displayer')
    def test_progress_forced_updates_bypass_debouncing(self, mock_displayer):
        """Test that forced updates bypass debouncing."""
        mock_display = Mock()
        mock_displayer.return_value = mock_display

        pbar = Progress(total=3, description="Forced", min_interval_sec=0.1)

        # Multiple forced updates should all go through
        initial_count = mock_display.call_count
        pbar._display(force=True)
        pbar._display(force=True)
        pbar._display(force=True)

        # Should have made 3 additional calls
        assert mock_display.call_count == initial_count + 3

        pbar.close()

    @pytest.mark.asyncio
    @patch('utils.progress.displayer')
    async def test_progress_in_async_context(self, mock_displayer):
        """Test progress bar works properly in async context."""
        mock_display = Mock()
        mock_displayer.return_value = mock_display

        # In async context, debouncing should work better
        with Progress(total=5, description="Async", min_interval_sec=0.02) as pbar:
            for _i in range(5):
                await asyncio.sleep(0.01)  # Async delay
                pbar.update(1)

        # Should have made display calls
        assert mock_display.call_count >= 1

    @patch('utils.progress.displayer')
    def test_progress_backwards_compatibility(self, mock_displayer):
        """Test that existing Progress API still works."""
        mock_display = Mock()
        mock_displayer.return_value = mock_display

        # All existing functionality should work
        pbar = Progress(total=3, description="Compat")

        # Update with different parameters
        pbar.update(1, suffix="Step 1")
        pbar.update(1, metrics={"loss": 0.5})
        pbar.update(1, suffix="Done", metrics={"loss": 0.1, "acc": 0.9})

        # Should work as before
        assert pbar.count == 3
        assert pbar.suffix == "Done"
        assert pbar.metrics["loss"] == 0.1
        assert pbar.metrics["acc"] == 0.9

        pbar.close()

    @patch('utils.progress.displayer')
    def test_progress_iterator_functionality(self, mock_displayer):
        """Test progress bar as iterator still works."""
        mock_display = Mock()
        mock_displayer.return_value = mock_display

        items = [1, 2, 3, 4, 5]
        results = []

        with Progress(items, description="Iterator") as pbar:
            for item in pbar:
                results.append(item)

        assert results == items
        assert mock_display.call_count >= 1

    @patch('utils.progress.displayer')
    def test_progress_context_manager(self, mock_displayer):
        """Test progress bar context manager functionality."""
        mock_display = Mock()
        mock_displayer.return_value = mock_display

        with Progress(total=2, description="Context") as pbar:
            pbar.update(1)
            pbar.update(1)
            # Should auto-close on exit

        assert pbar._closed

    @patch('utils.progress.displayer')
    def test_progress_min_interval_parameter(self, mock_displayer):
        """Test that min_interval_sec parameter is properly used."""
        mock_display = Mock()
        mock_displayer.return_value = mock_display

        # Test different intervals
        pbar1 = Progress(total=1, min_interval_sec=0.1)
        pbar2 = Progress(total=1, min_interval_sec=0.05)

        assert pbar1.min_interval_sec == 0.1
        assert pbar2.min_interval_sec == 0.05

        pbar1.close()
        pbar2.close()
