import logging
import sys
from typing import override

from lightning.pytorch.callbacks.progress.progress_bar import ProgressBar

from .progress import SyncProgress


class _ProgressLoggingHandler(logging.Handler):
    """A logging handler that uses the progress bar's print method to avoid display conflicts."""

    def __init__(self, progress_bar: 'LightningProgress'):
        super().__init__()
        self.progress_bar = progress_bar

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record using the progress bar's print method."""
        try:
            msg = self.format(record)
            self.progress_bar.print(msg)
        except Exception:
            self.handleError(record)


class LightningProgress(ProgressBar):
    """A progress bar that outputs basic text or HTML, with one bar for the whole process."""

    def __init__(self, interval: float = 0.3, install_logging_handler: bool = False):
        super().__init__()
        self._progress: SyncProgress | None = None
        self._enabled = True
        self.interval = interval
        """Minimum time between redraws (seconds)"""
        self.install_logging_handler = install_logging_handler
        """Whether to install a logging handler that uses the progress bar's print method"""
        self._logging_handler: _ProgressLoggingHandler | None = None
        self._original_console_handlers: list[logging.Handler] = []

    @override
    def disable(self) -> None:
        self._enabled = False

    @override
    def enable(self) -> None:
        self._enabled = True

    def is_enabled(self):
        # Called by the trainer (undocumented?)
        return self.progress is not None

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value: SyncProgress | None):
        if self._progress:
            self._progress.close()
        self._progress = value

    def _is_console_handler(self, handler: logging.Handler) -> bool:
        """Check if a logging handler writes to console (stdout or stderr)."""
        # Check if this is a StreamHandler and if its stream is stdout or stderr
        if isinstance(handler, logging.StreamHandler) and hasattr(handler, 'stream'):
            return handler.stream in (sys.stdout, sys.stderr)
        return False

    def _install_logging_handler(self) -> None:
        """Install a logging handler that uses the progress bar's print method."""
        if not self.install_logging_handler or self._logging_handler is not None:
            return

        # Create and configure the handler
        self._logging_handler = _ProgressLoggingHandler(self)

        # Find and temporarily remove console handlers from the root logger
        root_logger = logging.getLogger()
        console_handlers = [h for h in root_logger.handlers if self._is_console_handler(h)]

        for handler in console_handlers:
            root_logger.removeHandler(handler)
            self._original_console_handlers.append(handler)

        # Install our handler
        root_logger.addHandler(self._logging_handler)

    def _remove_logging_handler(self) -> None:
        """Remove the logging handler and restore original console handlers."""
        if self._logging_handler is None:
            return

        root_logger = logging.getLogger()

        # Remove our handler
        root_logger.removeHandler(self._logging_handler)
        self._logging_handler = None

        # Restore original console handlers
        for handler in self._original_console_handlers:
            root_logger.addHandler(handler)
        self._original_console_handlers.clear()

    def print(self, *args, **kwargs):
        if self.progress:
            self.progress.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    #
    # Training
    #

    @override
    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        if self._enabled:
            self.progress = SyncProgress(
                total=_resolve_total(trainer.estimated_stepping_batches),
                interval=self.interval,
            )
            self._install_logging_handler()

    @override
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if self.progress:
            self.progress.description = self.train_description
            self.progress.mark(self.get_prime_metric())

    def get_prime_metric(self):
        if not self.progress:
            return ''

        metric = self.progress.metrics.get('val_loss')
        if metric is None:
            metric = self.progress.metrics.get('train_loss')

        if isinstance(metric, float):
            return f'{metric:.4f}'
        elif metric is not None:
            return str(metric)
        else:
            return ''

    @override
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.progress:
            self.progress.count += 1
            self.progress.metrics |= self.get_metrics(trainer, pl_module)

    # @override
    # def on_train_epoch_end(self, trainer, pl_module):
    #     super().on_train_epoch_end(trainer, pl_module)
    #     if self.progress:
    #         self.progress.mark('')

    @override
    def on_validation_epoch_start(self, trainer, pl_module):
        # This happens DURING training, so we don't want to remove the existing progress bar.
        super().on_validation_epoch_start(trainer, pl_module)
        if self.progress:
            self.progress.description = self.validation_description

    @override
    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if self.progress:
            self.progress.metrics |= self.get_metrics(trainer, pl_module)

    @override
    def on_fit_end(self, trainer, pl_module):
        super().on_fit_end(trainer, pl_module)
        if self.progress:
            self.progress.mark(self.get_prime_metric())
        self._remove_logging_handler()
        self.progress = None

    #
    # Test
    #

    @override
    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        if self._enabled:
            self.progress = SyncProgress(
                total=_resolve_total(sum(trainer.num_test_batches)),
                interval=self.interval,
            )
            self._install_logging_handler()

    @override
    def on_test_epoch_start(self, trainer, pl_module):
        super().on_test_epoch_start(trainer, pl_module)
        if self.progress:
            self.progress.description = self.test_description

    @override
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.progress:
            self.progress.count += 1

    @override
    def on_test_epoch_end(self, trainer, pl_module):
        super().on_test_epoch_end(trainer, pl_module)
        if self.progress:
            self.progress.metrics |= self.get_metrics(trainer, pl_module)

    @override
    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        self._remove_logging_handler()
        self.progress = None

    # Predict

    @override
    def on_predict_start(self, trainer, pl_module):
        super().on_predict_start(trainer, pl_module)
        if self._enabled:
            self.progress = SyncProgress(
                total=_resolve_total(sum(trainer.num_predict_batches)),
                interval=self.interval,
            )
            self._install_logging_handler()

    @override
    def on_predict_epoch_start(self, trainer, pl_module):
        super().on_predict_epoch_start(trainer, pl_module)
        if self.progress:
            self.progress.description = self.predict_description

    @override
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.progress:
            self.progress.count += 1

    @override
    def on_predict_end(self, trainer, pl_module):
        super().on_predict_end(trainer, pl_module)
        self._remove_logging_handler()
        self.progress = None


def _resolve_total(total: int | float):
    """Normalize the total number of steps, because Lightning returns inf in some situations."""
    return int(total) if total != float('inf') else 0
