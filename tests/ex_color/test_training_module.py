"""Tests for the generalized TrainingModule."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from ex_color.model import ColorMLP, TrainingModule
from ex_color.regularizers.regularizer import RegularizerConfig
from mini.temporal.dopesheet import Dopesheet


class MockRegularizer:
    """Mock regularizer for testing."""

    def __call__(self, activations: Tensor) -> Tensor:
        # Simple regularizer that returns the mean squared values
        return (activations**2).mean(dim=-1)


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return ColorMLP()


@pytest.fixture
def sample_dopesheet():
    """Create a minimal dopesheet for testing."""
    return Dopesheet.from_csv(Path(__file__).parent / 'fixtures' / 'dopesheet.csv')


@pytest.fixture
def sample_objective():
    """Simple MSE objective for testing."""
    return torch.nn.MSELoss()


def test_training_module_with_layer_affinities(sample_model, sample_dopesheet, sample_objective):
    """Test that TrainingModule creates hooks based on layer_affinities."""
    regularizers = [
        RegularizerConfig(
            name='reg-encoder',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            layer_affinities=['encoder'],
        ),
        RegularizerConfig(
            name='reg-decoder',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            layer_affinities=['decoder'],
        ),
    ]

    training_module = TrainingModule(sample_model, sample_dopesheet, sample_objective, regularizers)
    training_module.on_fit_start()  # Set up hooks

    # Check that hooks were registered for both layers
    assert 'encoder' in training_module.latent_hooks
    assert 'decoder' in training_module.latent_hooks
    assert len(training_module.latent_hooks) == 2


def test_training_module_with_multiple_layers_per_regularizer(sample_model, sample_dopesheet, sample_objective):
    """Test that a regularizer can be applied to multiple layers."""
    regularizers = [
        RegularizerConfig(
            name='reg-multi',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            layer_affinities=['encoder', 'decoder'],
        ),
    ]

    training_module = TrainingModule(sample_model, sample_dopesheet, sample_objective, regularizers)
    training_module.on_fit_start()  # Set up hooks

    # Should register hooks for both layers
    assert 'encoder' in training_module.latent_hooks
    assert 'decoder' in training_module.latent_hooks
    assert len(training_module.latent_hooks) == 2


def test_training_module_invalid_layer_names(sample_model, sample_dopesheet, sample_objective):
    """Test that TrainingModule raises an exception for invalid layer names."""
    regularizers = [
        RegularizerConfig(
            name='reg-invalid',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            layer_affinities=['nonexistent_layer'],
        ),
    ]

    # Should raise an exception when hooks are set up
    training_module = TrainingModule(sample_model, sample_dopesheet, sample_objective, regularizers)
    with pytest.raises(AttributeError, match='Layer nonexistent_layer not found in model'):
        training_module.on_fit_start()


def test_training_step_with_layer_affinities(sample_model, sample_dopesheet, sample_objective):
    """Test that training_step correctly applies regularizers to specified layers."""
    regularizers = [
        RegularizerConfig(
            name='reg-encoder',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            layer_affinities=['encoder'],
        ),
    ]

    training_module = TrainingModule(sample_model, sample_dopesheet, sample_objective, regularizers)
    training_module.on_fit_start()  # Set up hooks
    training_module.log = MagicMock()  # Requires a Trainer, which this test doesn't use

    # Create sample batch
    batch_size = 4
    batch_data = torch.randn(batch_size, 3)
    batch_labels = {}
    batch = (batch_data, batch_labels)

    # Run training step
    result = training_module.training_step(batch, 0)

    # Should return loss and losses dict
    assert 'loss' in result
    assert 'losses' in result
    assert isinstance(result['loss'], torch.Tensor)
    assert isinstance(result['losses'], dict)

    # Should have reconstruction loss and regularizer loss
    assert 'recon' in result['losses']
    assert 'reg-encoder' in result['losses']


def test_hook_cleanup_on_fit_end(sample_model, sample_dopesheet, sample_objective):
    """Test that hooks are properly cleaned up in on_fit_end."""
    regularizers = [
        RegularizerConfig(
            name='reg-encoder',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            layer_affinities=['encoder'],
        ),
    ]

    training_module = TrainingModule(sample_model, sample_dopesheet, sample_objective, regularizers)

    # Initially, no hooks should be registered
    assert len(training_module.latent_hooks) == 0
    assert len(training_module.hook_handles) == 0

    # Set up hooks
    training_module.on_fit_start()
    assert len(training_module.latent_hooks) == 1
    assert len(training_module.hook_handles) == 1
    assert 'encoder' in training_module.latent_hooks
    assert 'encoder' in training_module.hook_handles

    # Clean up hooks
    training_module.on_fit_end()
    assert len(training_module.latent_hooks) == 0
    assert len(training_module.hook_handles) == 0
