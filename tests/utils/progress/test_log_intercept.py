"""High-value tests for the LogInterceptor and ProgressAwareLogHandler."""

import logging
import sys
from unittest.mock import Mock

import pytest

from utils.progress._log_intercept import LogInterceptor, ProgressAwareLogHandler

from .dummy_progress import DummyProgress


class DummyHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream or sys.stdout)
        self.records = []

    def emit(self, record):
        self.records.append(record)
        super().emit(record)


def test_progress_aware_log_handler_calls_progress_methods():
    progress = DummyProgress()
    handler = DummyHandler(stream=sys.stdout)
    wrapper = ProgressAwareLogHandler(progress, handler)
    record = logging.LogRecord('test', logging.INFO, '', 0, 'msg', (), None)
    wrapper.emit(record)
    assert progress.calls == ['before', 'after']
    assert handler.records == [record]


def test_progress_aware_log_handler_handles_exceptions():
    progress = DummyProgress()
    handler = DummyHandler(stream=sys.stdout)
    wrapper = ProgressAwareLogHandler(progress, handler)

    # Patch handler.handle to raise
    def raise_exc(record):
        raise RuntimeError('fail')

    handler.handle = raise_exc
    # Patch handleError to track call
    wrapper.handleError = Mock()
    record = logging.LogRecord('test', logging.INFO, '', 0, 'msg', (), None)
    wrapper.emit(record)
    wrapper.handleError.assert_called_once_with(record)


def test_log_interceptor_start_and_stop_restores_handlers(monkeypatch):
    # Set up a logger with a console handler
    logger = logging.getLogger('logintercept-test')
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    progress = DummyProgress()
    interceptor = LogInterceptor()
    # Start interception
    interceptor.start(progress)
    # The original handler should be replaced
    assert all(not isinstance(h, type(handler)) or isinstance(h, ProgressAwareLogHandler) for h in logger.handlers)
    # Stop interception
    interceptor.stop()
    # The original handler should be restored
    assert any(isinstance(h, type(handler)) for h in logger.handlers)


def test_log_interceptor_double_start_raises():
    interceptor = LogInterceptor()
    progress = DummyProgress()
    interceptor.intercepted_handlers = [('foo', Mock(), Mock())]
    with pytest.raises(RuntimeError):
        interceptor.start(progress)


def test_log_interceptor_stop_is_idempotent():
    interceptor = LogInterceptor()
    # Should not raise even if nothing to stop
    interceptor.stop()
