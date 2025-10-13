"""Tests for the workflows module."""

import warnings

import numpy as np
import torch

from ex_color.data.color_cube import ColorCube
from ex_color.workflows import evaluate_model_on_cube, infer_with_latent_capture


def _ignore_dl_warnings():
    # Silence Lightning's suggestion about low num_workers in predict_dataloader for tests.
    warnings.filterwarnings(
        'ignore',
        message=r'.*predict_dataloader.*does not have many workers.*',
    )


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

    with warnings.catch_warnings():
        _ignore_dl_warnings()
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

    with warnings.catch_warnings():
        _ignore_dl_warnings()
        result = evaluate_model_on_cube(model, interventions=[], test_data=cube)

    assert 'recon' in result.vars
    assert 'MSE' in result.vars
    assert 'latents' in result.vars
    assert result['recon'].shape == (*cube.shape, 3)
    assert result['MSE'].shape == (*cube.shape, 1)
    assert result['latents'].shape[:-1] == cube.shape  # Last dimension is latent dim
