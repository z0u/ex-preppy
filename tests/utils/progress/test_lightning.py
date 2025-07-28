"""Test the LightningProgress logging handler functionality."""

import logging
import sys
from io import StringIO
from unittest.mock import Mock, patch

import pytest
from utils.progress.lightning import LightningProgress, _ProgressLoggingHandler


class TestProgressLoggingHandler:
    """Test the _ProgressLoggingHandler."""

    @pytest.fixture
    def mock_progress_bar(self):
        """Return a mock LightningProgress instance."""
        mock_bar = Mock(spec=LightningProgress)
        return mock_bar

    @pytest.fixture
    def handler(self, mock_progress_bar):
        """Return a _ProgressLoggingHandler instance."""
        return _ProgressLoggingHandler(mock_progress_bar)

    def test_emit_calls_progress_bar_print(self, handler, mock_progress_bar):
        """Test that emit() calls the progress bar's print method."""
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        mock_progress_bar.print.assert_called_once_with('Test message')

    def test_emit_handles_exceptions(self, handler, mock_progress_bar):
        """Test that emit() handles exceptions properly."""
        # Make the print method raise an exception
        mock_progress_bar.print.side_effect = RuntimeError('Test error')

        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test message',
            args=(),
            exc_info=None,
        )

        # Mock the handleError method to track if it's called
        handler.handleError = Mock()

        handler.emit(record)

        handler.handleError.assert_called_once_with(record)


class TestLightningProgress:
    """Test the LightningProgress class with logging handler functionality."""

    def test_init_default_no_logging_handler(self):
        """Test that logging handler is not installed by default."""
        progress = LightningProgress()
        assert progress.install_logging_handler is False
        assert progress._logging_handler is None
        assert progress._original_console_handlers == []

    def test_init_with_logging_handler_enabled(self):
        """Test initialization with logging handler enabled."""
        progress = LightningProgress(install_logging_handler=True)
        assert progress.install_logging_handler is True
        assert progress._logging_handler is None  # Not installed until progress starts
        assert progress._original_console_handlers == []

    def test_is_console_handler_with_stream_handler(self):
        """Test _is_console_handler with StreamHandler."""
        progress = LightningProgress()

        # Test with stdout stream
        stdout_handler = logging.StreamHandler(sys.stdout)
        assert progress._is_console_handler(stdout_handler) is True

        # Test with stderr stream
        stderr_handler = logging.StreamHandler(sys.stderr)
        assert progress._is_console_handler(stderr_handler) is True

        # Test with other stream
        string_io = StringIO()
        file_handler = logging.StreamHandler(string_io)
        assert progress._is_console_handler(file_handler) is False

    def test_is_console_handler_with_file_handler(self):
        """Test _is_console_handler with FileHandler."""
        progress = LightningProgress()

        # FileHandler doesn't have a stream attribute in the same way
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = '/tmp/test.log'
            file_handler = logging.FileHandler('/tmp/test.log')
            assert progress._is_console_handler(file_handler) is False

    def test_install_logging_handler_disabled(self):
        """Test that logging handler is not installed when disabled."""
        progress = LightningProgress(install_logging_handler=False)

        with patch('logging.getLogger') as mock_get_logger:
            progress._install_logging_handler()
            mock_get_logger.assert_not_called()

    def test_install_logging_handler_enabled(self):
        """Test logging handler installation when enabled."""
        progress = LightningProgress(install_logging_handler=True)

        # Create mock handlers
        mock_console_handler = Mock(spec=logging.StreamHandler)
        mock_console_handler.stream = sys.stdout
        mock_file_handler = Mock(spec=logging.FileHandler)

        # Mock the root logger
        mock_root_logger = Mock()
        mock_root_logger.handlers = [mock_console_handler, mock_file_handler]

        with patch('logging.getLogger', return_value=mock_root_logger):
            with patch.object(progress, '_is_console_handler') as mock_is_console:
                mock_is_console.side_effect = lambda h: h == mock_console_handler

                progress._install_logging_handler()

                # Verify our handler was added
                assert progress._logging_handler is not None
                mock_root_logger.addHandler.assert_called_once_with(progress._logging_handler)

                # Verify console handler was removed
                mock_root_logger.removeHandler.assert_called_once_with(mock_console_handler)
                assert mock_console_handler in progress._original_console_handlers

                # Verify file handler was not touched
                assert mock_file_handler not in progress._original_console_handlers

    def test_remove_logging_handler(self):
        """Test logging handler removal."""
        progress = LightningProgress(install_logging_handler=True)

        # Set up some state as if handler was installed
        mock_handler = Mock()
        mock_console_handler = Mock()
        progress._logging_handler = mock_handler
        progress._original_console_handlers = [mock_console_handler]

        mock_root_logger = Mock()

        with patch('logging.getLogger', return_value=mock_root_logger):
            progress._remove_logging_handler()

            # Verify our handler was removed
            mock_root_logger.removeHandler.assert_called_once_with(mock_handler)
            assert progress._logging_handler is None

            # Verify original handler was restored
            mock_root_logger.addHandler.assert_called_once_with(mock_console_handler)
            assert progress._original_console_handlers == []

    def test_remove_logging_handler_when_none_installed(self):
        """Test that removing handler when none is installed does nothing."""
        progress = LightningProgress()

        with patch('logging.getLogger') as mock_get_logger:
            progress._remove_logging_handler()
            mock_get_logger.assert_not_called()

    def test_print_delegates_to_progress(self):
        """Test that print method delegates to progress when available."""
        progress = LightningProgress()

        # Mock the progress
        mock_sync_progress = Mock()
        progress._progress = mock_sync_progress

        progress.print('test message', flush=True)

        mock_sync_progress.print.assert_called_once_with('test message', flush=True)

    def test_print_falls_back_to_builtin_print(self):
        """Test that print method falls back to built-in print when no progress."""
        progress = LightningProgress()
        progress._progress = None

        with patch('builtins.print') as mock_print:
            progress.print('test message', flush=True)
            mock_print.assert_called_once_with('test message', flush=True)
