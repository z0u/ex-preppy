from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch import Callback, Trainer
from torch import Tensor

log = logging.getLogger(__name__)


class LabelProportionCallback(Callback):
    """
    Aggregate and log the proportion of samples receiving each label.

    Assumptions:
    - Each training batch is a tuple of (data: Tensor, labels: dict[str, Tensor]).
    - Label tensors are shaped [B] with values in [0, 1], typically 0/1.
    - Proportion for a label in a batch is sum(label_tensor) / len(label_tensor).

    Logging:
    - At epoch end: logs per-epoch proportions under "labels/epoch/{label}".
    - At fit end: logs global proportions under "labels/total/{label}".

    Distributed:
    - Reductions are performed via the strategy (sums across devices/processes).

    Usage:
    >>> from ex_color.callbacks import LabelProportionCallback
    >>> cbs = [LabelProportionCallback(prefix="labels")]  # add to Trainer(callbacks=cbs)
    """

    def __init__(self, *, prefix: str = 'labels') -> None:
        super().__init__()
        self.prefix = prefix
        self._epoch_label_sums: Dict[str, float] = defaultdict(float)
        self._epoch_counts: int = 0
        self._total_label_sums: Dict[str, float] = defaultdict(float)
        self._total_counts: int = 0

    # ---- helpers ----
    def _accumulate_batch(self, trainer: Trainer, labels: dict[str, Tensor]):
        if not labels:
            return
        # Assume all labels tensors share batch dimension size
        # Count samples from the first label tensor
        any_label = next(iter(labels.values()))
        batch_size_local = int(any_label.shape[0])

        # Device-safe: ensure tensors are float32 on the current device
        # Reduce sums across processes/devices
        label_sums = {}
        for name, t in labels.items():
            t = t.detach()
            # If label is not on the same device as strategy, move it for reduction
            if t.dtype not in (torch.float32, torch.float64):
                t = t.float()
            sum_local = t.sum()
            sum_global = trainer.strategy.reduce(sum_local, reduce_op='sum')  # type: ignore[arg-type]
            label_sums[name] = float(sum_global.item())

        # Global batch size across devices
        count_local = torch.tensor(batch_size_local, device=any_label.device, dtype=torch.float32)
        count_global = trainer.strategy.reduce(count_local, reduce_op='sum')  # type: ignore[arg-type]
        batch_count_global = int(count_global.item())

        # Update epoch and total accumulators
        for name, s in label_sums.items():
            self._epoch_label_sums[name] += s
            self._total_label_sums[name] += s
        self._epoch_counts += batch_count_global
        self._total_counts += batch_count_global

    def _log_dict(self, trainer: Trainer, values: Dict[str, float], *, step: int | None = None):
        logger = trainer.logger
        if logger is None:
            return
        logger.log_metrics(values, step=step if step is not None else trainer.global_step)

    # ---- Lightning hooks ----
    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx: int) -> None:  # noqa: ANN001
        # batch is expected to be (data, labels)
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            return
        labels = batch[1]
        if not isinstance(labels, dict):
            return
        self._accumulate_batch(trainer, labels)

    def on_train_epoch_start(self, trainer: Trainer, pl_module) -> None:  # noqa: ANN001
        # Reset epoch accumulators
        self._epoch_label_sums.clear()
        self._epoch_counts = 0

    def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:  # noqa: ANN001
        # Compute and log per-epoch proportions
        if self._epoch_counts <= 0:
            return
        metrics_ = {name: s / max(1, self._total_counts) for name, s in sorted(self._total_label_sums.items())}
        metrics = {f'{self.prefix}/epoch/{name}': v for name, v in metrics_.items()}

        # Only rank zero triggers the logger call to avoid duplicates
        @rank_zero_only
        def _log():
            if trainer.logger:
                trainer.logger.log_metrics(metrics, step=trainer.global_step)

        _log()

    def on_fit_end(self, trainer: Trainer, pl_module) -> None:  # noqa: ANN001
        # Log total proportions over the entire training
        if self._total_counts <= 0:
            return
        metrics_ = {name: s / max(1, self._total_counts) for name, s in sorted(self._total_label_sums.items())}
        metrics = {f'{self.prefix}/{name}': v for name, v in metrics_.items()}

        @rank_zero_only
        def _log():
            if trainer.logger:
                trainer.logger.log_metrics(metrics)
            human_readable = [f'{k}: {v:.2%}' for k, v in metrics_.items()]
            log.info('Label frequencies: %s', ', '.join(human_readable))

        _log()
