"""Tests for the workflows module."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from ex_color.data.color_cube import ColorCube
from ex_color.workflows import (
    evaluate_model_on_cube,
    infer_with_latent_capture,
    prep_train_data,
    prep_val_data,
)


def test_prep_train_data():
    """Test prep_train_data creates a valid DataLoader."""
    loader = prep_train_data(training_subs=3, batch_size=8)

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 8

    # Check that we can get a batch
    batch = next(iter(loader))
    colors, labels = batch

    assert isinstance(colors, torch.Tensor)
    assert colors.shape[0] == 8  # batch size
    assert colors.shape[1] == 3  # RGB
    assert isinstance(labels, dict)


def test_prep_train_data_custom_weight():
    """Test prep_train_data with custom red weight function."""

    def custom_weight(c):
        return np.ones_like(c[..., 0]) * 0.5

    loader = prep_train_data(training_subs=3, batch_size=8, red_weight_fn=custom_weight)

    assert isinstance(loader, DataLoader)

    # Check that we can get a batch
    batch = next(iter(loader))
    colors, labels = batch

    assert isinstance(colors, torch.Tensor)
    assert colors.shape[0] == 8


def test_prep_val_data():
    """Test prep_val_data creates a valid DataLoader."""
    loader = prep_val_data(training_subs=3, batch_size=8)

    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 8

    # Check that we can get a batch
    batch = next(iter(loader))
    colors, labels = batch

    assert isinstance(colors, torch.Tensor)
    assert colors.shape[0] == 8  # batch size
    assert colors.shape[1] == 3  # RGB
    assert isinstance(labels, dict)


def test_prep_val_data_custom_filter():
    """Test prep_val_data with custom red filter function."""

    def custom_filter(c):
        return np.ones_like(c[..., 0], dtype=bool)

    loader = prep_val_data(training_subs=3, batch_size=8, red_filter_fn=custom_filter)

    assert isinstance(loader, DataLoader)

    # Check that we can get a batch
    batch = next(iter(loader))
    colors, labels = batch

    assert isinstance(colors, torch.Tensor)
    assert colors.shape[0] == 8


def test_infer_with_latent_capture():
    """Test infer_with_latent_capture runs inference and captures latents."""

    # Create a simple model with a bottleneck
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 4)
            self.bottleneck = torch.nn.Identity()
            self.decoder = torch.nn.Linear(4, 3)

        def forward(self, x):
            x = self.encoder(x)
            x = self.bottleneck(x)
            x = self.decoder(x)
            return x

    model = SimpleModel()
    test_data = torch.rand(2, 3, 3)  # 2x3 grid of RGB colors

    predictions, latents = infer_with_latent_capture(model, test_data, interventions=[], layer_name='bottleneck')

    assert predictions.shape == test_data.shape
    assert latents.shape[0] == 2 * 3  # flattened
    assert latents.shape[1] == 4  # bottleneck dimension


def test_test_model_on_cube():
    """Test test_model_on_cube adds reconstruction and loss to ColorCube."""

    # Create a simple model with a bottleneck
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 4)
            self.bottleneck = torch.nn.Identity()
            self.decoder = torch.nn.Linear(4, 3)

        def forward(self, x):
            x = self.encoder(x)
            x = self.bottleneck(x)
            x = self.decoder(x)
            return x

    model = SimpleModel()
    cube = ColorCube.from_rgb(r=np.linspace(0, 1, 2), g=np.linspace(0, 1, 2), b=np.linspace(0, 1, 2))

    result = evaluate_model_on_cube(model, interventions=[], test_data=cube)

    assert 'recon' in result.vars
    assert 'MSE' in result.vars
    assert 'latents' in result.vars
    assert result['recon'].shape == (*cube.shape, 3)
    assert result['MSE'].shape == (*cube.shape, 1)
    assert result['latents'].shape[:-1] == cube.shape  # Last dimension is latent dim
