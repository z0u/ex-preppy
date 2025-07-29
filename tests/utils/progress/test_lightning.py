"""Test the LightningProgress logging handler functionality."""

from unittest.mock import Mock, patch

import pytest
from utils.progress.lightning import LightningProgress


# The ProgressAwareLogHandler is now internal; test interception via LogInterceptor


@pytest.fixture
def mock_log_interceptor(monkeypatch):
    """Patch LogInterceptor on LightningProgress and yield a mock instance."""
    from utils.progress import lightning

    mock = Mock()
    monkeypatch.setattr(lightning, 'LogInterceptor', lambda: mock)
    yield mock


class TestLightningProgress:
    """Test the LightningProgress class with logging handler functionality."""

    def test_print_delegates_to_progress(self):
        progress = LightningProgress()
        mock_sync_progress = Mock()
        progress._progress = mock_sync_progress
        progress.print('test message', flush=True)
        mock_sync_progress.print.assert_called_once_with('test message', flush=True)

    def test_print_falls_back_to_builtin_print(self):
        progress = LightningProgress()
        progress._progress = None
        with patch('builtins.print') as mock_print:
            progress.print('test message', flush=True)
            mock_print.assert_called_once_with('test message', flush=True)

    def test_log_interceptor_start_and_stop(self, mock_log_interceptor):
        progress = LightningProgress(install_logging_handler=True)
        progress._enabled = True
        progress.progress = Mock()
        # Simulate start of fit
        mock_log_interceptor.start.reset_mock()
        mock_log_interceptor.stop.reset_mock()
        progress._log_interceptor.start(progress.progress)
        mock_log_interceptor.start.assert_called_once_with(progress.progress)
        # Simulate end of fit
        progress._log_interceptor.stop()
        mock_log_interceptor.stop.assert_called_once()
