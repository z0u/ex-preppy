"""Tests for LabelProportionCallback."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import SingleDeviceStrategy

from ex_color.callbacks import LabelProportionCallback


@pytest.fixture
def mock_trainer():
    """Create a mock trainer with minimal required attributes."""
    trainer = Mock(spec=Trainer)
    trainer.strategy = SingleDeviceStrategy(device=torch.device('cpu'))
    trainer.global_step = 0
    trainer.logger = None
    return trainer


def test_accumulate_batch_with_labels(mock_trainer):
    """Test that batches with labels are counted correctly."""
    callback = LabelProportionCallback(prefix='test', get_active_labels=None)

    # Create sample batch with labels
    # Layout: first 5 samples have both red+blue, next 5 have only red, next 5 have only blue, rest have nothing
    batch_size = 32
    labels = {
        'red': torch.tensor([1.0] * 10 + [0.0] * 22),  # 10 red samples (indices 0-9)
        'blue': torch.tensor([1.0] * 5 + [0.0] * 5 + [1.0] * 5 + [0.0] * 17),  # 10 blue samples (indices 0-4, 10-14)
    }

    callback._accumulate_batch(mock_trainer, batch_size, labels)

    assert callback._total_counts == batch_size
    assert callback._total_label_sums['red'] == pytest.approx(10.0)
    assert callback._total_label_sums['blue'] == pytest.approx(10.0)
    # _any should count samples with at least one label: 0-4 (both), 5-9 (red only), 10-14 (blue only) = 15 total
    assert callback._total_label_sums['_any'] == pytest.approx(15.0)


def test_accumulate_batch_without_labels(mock_trainer):
    """Test that batches without active labels are still counted."""
    callback = LabelProportionCallback(prefix='test', get_active_labels=None)

    # Accumulate a batch with no labels (empty dict)
    batch_size = 64
    callback._accumulate_batch(mock_trainer, batch_size, {})

    # The batch should still be counted even though there are no labels
    assert callback._total_counts == batch_size
    assert len(callback._total_label_sums) == 0  # No labels accumulated


def test_accumulate_multiple_batches_mixed(mock_trainer):
    """Test accumulating multiple batches, some with and some without labels."""
    callback = LabelProportionCallback(prefix='test', get_active_labels=None)

    # First batch: with labels
    callback._accumulate_batch(
        mock_trainer,
        64,
        {'red': torch.tensor([1.0] * 5 + [0.0] * 59)},
    )

    # Second batch: no labels (e.g., regularizers are off)
    callback._accumulate_batch(mock_trainer, 64, {})

    # Third batch: with labels again
    callback._accumulate_batch(
        mock_trainer,
        64,
        {'red': torch.tensor([1.0] * 3 + [0.0] * 61)},
    )

    # All three batches should be counted
    assert callback._total_counts == 64 * 3
    # Only the labeled batches contribute to label sums
    assert callback._total_label_sums['red'] == pytest.approx(8.0)
    assert callback._total_label_sums['_any'] == pytest.approx(8.0)


def test_on_train_batch_end_filters_inactive_labels(mock_trainer):
    """Test that on_train_batch_end correctly filters out inactive labels."""
    active_labels = {'red'}  # Only 'red' is active
    callback = LabelProportionCallback(prefix='test', get_active_labels=lambda: active_labels)

    # Create a batch with both red and blue labels
    data = torch.randn(32, 3)
    labels = {
        'red': torch.tensor([1.0] * 5 + [0.0] * 27),
        'blue': torch.tensor([1.0] * 10 + [0.0] * 22),  # This should be filtered out
    }
    batch = (data, labels)

    callback.on_train_batch_end(mock_trainer, Mock(), None, batch, 0)  # type: ignore[arg-type]

    # Batch should be counted
    assert callback._total_counts == 32
    # Only 'red' should be accumulated (blue is filtered out)
    assert callback._total_label_sums['red'] == pytest.approx(5.0)
    assert 'blue' not in callback._total_label_sums
    assert callback._total_label_sums['_any'] == pytest.approx(5.0)


def test_on_train_batch_end_with_no_active_labels(mock_trainer):
    """Test that batches are counted even when no labels are active."""
    callback = LabelProportionCallback(prefix='test', get_active_labels=lambda: set())  # No active labels

    # Create a batch with labels
    data = torch.randn(64, 3)
    labels = {
        'red': torch.tensor([1.0] * 10 + [0.0] * 54),
        'blue': torch.tensor([1.0] * 10 + [0.0] * 54),
    }
    batch = (data, labels)

    callback.on_train_batch_end(mock_trainer, Mock(), None, batch, 0)  # type: ignore[arg-type]

    # Batch should still be counted even though all labels are filtered out
    assert callback._total_counts == 64
    # No labels should be accumulated since none are active
    assert len(callback._total_label_sums) == 0
