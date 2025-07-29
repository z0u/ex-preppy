import logging
import sys
from typing import override

from utils.progress.progress import ProgressBase


class ProgressAwareLogHandler(logging.Handler):
    """A logging handler that avoids display conflicts with the progress bar."""

    def __init__(self, progress: ProgressBase, handler: logging.Handler):
        super().__init__()
        self.progress = progress
        self.handler = handler

    @override
    def emit(self, record):
        try:
            self.progress.before_print()
            self.handler.handle(record)
            self.progress.after_print()
        except Exception:
            self.handleError(record)

    def __repr__(self):
        return f'{type(self).__name__}({self.handler!r})'


class LogInterceptor:
    """Prevents console progress bars from being messed up by loging."""

    def __init__(self):
        self.intercepted_handlers: list[tuple[str, logging.Handler, ProgressAwareLogHandler]] = []

    def start(self, progress: ProgressBase):
        if self.intercepted_handlers:
            raise RuntimeError('Already intercepted!')

        console_handlers = [
            (logger, handler)  #
            for logger in logging.Logger.manager.loggerDict.values()
            if isinstance(logger, logging.Logger)
            for handler in logger.handlers
            if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr)
        ]

        try:
            for logger, handler in console_handlers:
                wrapper = ProgressAwareLogHandler(progress, handler)
                logger.removeHandler(handler)
                try:
                    logger.addHandler(wrapper)
                except:
                    # Undo
                    logger.addHandler(handler)
                    logger.removeHandler(wrapper)  # Noop if the handler wasn't added
                    raise
                self.intercepted_handlers.append((logger.name, handler, wrapper))
        except:
            # Undo
            self.stop()
            raise

    def stop(self):
        while self.intercepted_handlers:
            name, handler, wrapper = self.intercepted_handlers.pop()
            try:
                logger = logging.getLogger(name)
                logger.removeHandler(wrapper)
                try:
                    logger.addHandler(handler)
                except:
                    # Undo
                    logger.addHandler(wrapper)
                    logger.removeHandler(handler)  # Noop if the handler wasn't added
                    raise
            except:
                # Undo
                self.intercepted_handlers.append((name, handler, wrapper))
                raise


def get_console_handlers():
    return [
        (logger, handler)  #
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr)
    ]
