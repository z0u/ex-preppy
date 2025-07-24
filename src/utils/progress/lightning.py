from typing import override

from lightning.pytorch.callbacks.progress.progress_bar import ProgressBar

from .progress import SyncProgress


class LightningProgress(ProgressBar):
    """A progress bar that outputs basic text or HTML, with one bar for the whole process."""

    def __init__(self, interval: float = 0.3):
        super().__init__()
        self._progress: SyncProgress | None = None
        self._enabled = True
        self.interval = interval
        """Minimum time between redraws (seconds)"""

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

    @override
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
        self.progress = None


def _resolve_total(total: int | float):
    """Normalize the total number of steps, because Lightning returns inf in some situations."""
    return int(total) if total != float('inf') else 0
