"""Tests for the generalized TrainingModule."""

import pytest
import torch
from torch import Tensor

from ex_color.model import TrainingModule, ColorMLP, ColorMLPTrainingModule
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
    return Dopesheet.from_csv('/tmp/test_dopesheet.csv')


@pytest.fixture
def sample_objective():
    """Simple MSE objective for testing."""
    return torch.nn.MSELoss()


def test_training_module_with_layer_affinities(sample_model, sample_dopesheet, sample_objective):
    """Test that TrainingModule correctly handles layer_affinities."""
    regularizers = [
        RegularizerConfig(
            name='test-reg-encoder',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            layer_affinities=['encoder'],
        ),
        RegularizerConfig(
            name='test-reg-decoder',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            layer_affinities=['decoder'],
        ),
    ]

    training_module = TrainingModule(
        model=sample_model, dopesheet=sample_dopesheet, objective=sample_objective, regularizers=regularizers
    )

    # Check that hooks were registered for both layers
    assert 'encoder' in training_module.latent_hooks
    assert 'decoder' in training_module.latent_hooks
    assert len(training_module.latent_hooks) == 2


def test_training_module_backwards_compatibility(sample_dopesheet, sample_objective):
    """Test that ColorMLPTrainingModule provides backwards compatibility for regularizers without layer_affinities."""
    regularizers = [
        RegularizerConfig(
            name='test-reg-old',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            # No layer_affinities - should work with ColorMLPTrainingModule but not TrainingModule
        ),
    ]

    # This should work with ColorMLPTrainingModule (backwards compatibility)
    training_module = ColorMLPTrainingModule(
        dopesheet=sample_dopesheet, objective=sample_objective, regularizers=regularizers
    )

    # Should default to encoder hook for backwards compatibility
    assert 'encoder' in training_module.latent_hooks
    assert len(training_module.latent_hooks) == 1

    # TrainingModule should raise an error for regularizers without layer_affinities
    with pytest.raises(ValueError, match='must specify layer_affinities'):
        TrainingModule(
            model=ColorMLP(), dopesheet=sample_dopesheet, objective=sample_objective, regularizers=regularizers
        ).training_step((torch.randn(4, 3), {}), 0)


def test_training_module_with_multiple_layers_per_regularizer(sample_model, sample_dopesheet, sample_objective):
    """Test that a regularizer can be applied to multiple layers."""
    regularizers = [
        RegularizerConfig(
            name='test-reg-multi',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            layer_affinities=['encoder', 'decoder'],
        ),
    ]

    training_module = TrainingModule(
        model=sample_model, dopesheet=sample_dopesheet, objective=sample_objective, regularizers=regularizers
    )

    # Should register hooks for both layers
    assert 'encoder' in training_module.latent_hooks
    assert 'decoder' in training_module.latent_hooks
    assert len(training_module.latent_hooks) == 2


def test_training_module_invalid_layer_names(sample_model, sample_dopesheet, sample_objective):
    """Test that TrainingModule raises an exception for invalid layer names."""
    regularizers = [
        RegularizerConfig(
            name='test-reg-invalid',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            layer_affinities=['nonexistent_layer'],
        ),
    ]

    # Should raise an exception during initialization
    with pytest.raises(AttributeError, match='Layer nonexistent_layer not found in model'):
        TrainingModule(
            model=sample_model, dopesheet=sample_dopesheet, objective=sample_objective, regularizers=regularizers
        )


def test_training_step_with_layer_affinities(sample_model, sample_dopesheet, sample_objective):
    """Test that training_step correctly applies regularizers to specified layers."""
    regularizers = [
        RegularizerConfig(
            name='test-reg-encoder',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            layer_affinities=['encoder'],
        ),
    ]

    training_module = TrainingModule(
        model=sample_model, dopesheet=sample_dopesheet, objective=sample_objective, regularizers=regularizers
    )

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
    assert 'test-reg-encoder' in result['losses']


def test_training_module_no_layer_affinities(sample_model, sample_dopesheet, sample_objective):
    """Test that TrainingModule creates no hooks when no regularizers specify layer_affinities."""
    regularizers = [
        RegularizerConfig(
            name='test-reg-no-layers',
            compute_loss_term=MockRegularizer(),
            label_affinities=None,
            # No layer_affinities specified
        ),
    ]

    training_module = TrainingModule(
        model=sample_model, dopesheet=sample_dopesheet, objective=sample_objective, regularizers=regularizers
    )

    # Should not register any hooks when no layer_affinities are specified
    assert len(training_module.latent_hooks) == 0
