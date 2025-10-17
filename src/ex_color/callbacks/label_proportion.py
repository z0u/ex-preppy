from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Collection, Dict, override

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

    def __init__(self, *, prefix: str = 'labels', get_active_labels: Callable[[], Collection[str]] | None):
        super().__init__()
        self.prefix = prefix
        self.get_active_labels = get_active_labels
        self._epoch_label_sums: Dict[str, float] = defaultdict(float)
        self._epoch_counts: int = 0
        self._total_label_sums: Dict[str, float] = defaultdict(float)
        self._total_counts: int = 0

    # ---- helpers ----
    def _accumulate_batch(self, trainer: Trainer, batch_size: int, labels: dict[str, Tensor]):
        """
        Accumulate label statistics for a batch.

        Args:
            trainer: The Lightning trainer.
            batch_size: The local batch size (number of samples in this batch on this device).
            labels: Dictionary of label tensors, may be empty if no labels are active.
        """
        # Get the device from trainer or default to CPU
        device = trainer.strategy.root_device if hasattr(trainer.strategy, 'root_device') else torch.device('cpu')

        # Global batch size across devices - always count this regardless of whether labels are active
        count_local = torch.tensor(batch_size, device=device, dtype=torch.float32)
        count_global = trainer.strategy.reduce(count_local, reduce_op='sum')  # type: ignore[arg-type]
        batch_count_global = int(count_global.item())

        # Update batch counts (always, even if no active labels)
        self._epoch_counts += batch_count_global
        self._total_counts += batch_count_global

        # If no labels are active, we're done (but we've still counted the batch)
        if not labels:
            return

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

        # Count samples with any label (at least one label > 0)
        # Stack all label tensors and check if any label is > 0 for each sample
        label_tensors = [t.detach().float() for t in labels.values()]
        stacked_labels = torch.stack(label_tensors, dim=1)  # [batch_size, num_labels]
        any_label_mask = (stacked_labels > 0).any(dim=1)  # [batch_size]
        any_label_count_local = any_label_mask.sum()
        any_label_count_global = trainer.strategy.reduce(any_label_count_local, reduce_op='sum')  # type: ignore[arg-type]
        any_label_count = float(any_label_count_global.item())

        # Update epoch and total accumulators
        for name, s in label_sums.items():
            self._epoch_label_sums[name] += s
            self._total_label_sums[name] += s
        self._epoch_label_sums['_any'] += any_label_count
        self._total_label_sums['_any'] += any_label_count

    def _log_dict(self, trainer: Trainer, values: Dict[str, float], *, step: int | None = None):
        logger = trainer.logger
        if logger is None:
            return
        logger.log_metrics(values, step=step if step is not None else trainer.global_step)

    # ---- Lightning hooks ----
    @override
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # batch is expected to be (data, labels)
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Batch isn't labelled. Add a collate function or remove this callback.")
        data = batch[0]
        labels = batch[1]
        if not isinstance(labels, dict):
            raise ValueError("Batch isn't labelled. Add a collate function or remove this callback.")

        # Get batch size from data tensor
        batch_size = int(data.shape[0])

        if self.get_active_labels:
            # Select labels that are used in this batch. E.g. according to which regularizers are on-schedule (active).
            active_labels = self.get_active_labels()
            labels = {k: v for k, v in labels.items() if k in active_labels}
        self._accumulate_batch(trainer, batch_size, labels)

    @override
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset epoch accumulators
        self._epoch_label_sums.clear()
        self._epoch_counts = 0

    @override
    def on_train_epoch_end(self, trainer, pl_module):
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

    @override
    def on_fit_end(self, trainer, pl_module):
        # Log total proportions over the entire training
        if self._total_counts <= 0:
            return
        # Counts
        metrics_n = {
            f'{self.prefix}/n/{name}': v  #
            for name, v in sorted(self._total_label_sums.items())
        }
        # Proportions
        metrics_p = {
            f'{self.prefix}/p/{name}': v / max(1, self._total_counts)  #
            for name, v in sorted(self._total_label_sums.items())
        }

        @rank_zero_only
        def _log():
            if trainer.logger:
                trainer.logger.log_metrics(metrics_n | metrics_p | {f'{self.prefix}/n_total': self._total_counts})
            # Format: label: count (percentage)
            human_readable = [f'{k}: {v}' for k, v in metrics_n.items()]
            log.info('Label frequencies (n=%d): %s', self._total_counts, ', '.join(human_readable))

        _log()
