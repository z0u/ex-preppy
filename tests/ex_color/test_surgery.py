import torch

from ex_color.model import CNColorMLP
from ex_color.surgery import prune, ablate
from ex_color.hooks import ActivationCaptureHook


def test_prune_reduces_latent_width_and_shapes():
    model = CNColorMLP(4)

    # Before
    enc_last: torch.nn.Linear = model.encoder[-1]  # type: ignore[index]
    dec_first: torch.nn.Linear = model.decoder[0]  # type: ignore[index]
    assert enc_last.out_features == 4
    assert dec_first.in_features == 4

    m2 = prune(model, 'bottleneck', [0])

    # After: shapes reduced
    enc_last2: torch.nn.Linear = m2.encoder[-1]  # type: ignore[index]
    dec_first2: torch.nn.Linear = m2.decoder[0]  # type: ignore[index]
    assert enc_last2.out_features == 3
    assert dec_first2.in_features == 3

    # Forward still works and outputs clamped in [0,1]
    x = torch.rand(5, 3)
    y = m2(x)
    assert y.shape == (5, 3)
    assert torch.isfinite(y).all()


def test_ablate_zeros_selected_bottleneck_dims():
    torch.manual_seed(0)
    model = CNColorMLP(4)

    # Zero-out dimension 0 of the bottleneck
    m2 = ablate(model, 'bottleneck', [0])

    # Hook to capture bottleneck activations
    hook = ActivationCaptureHook()
    handle = m2.get_submodule('bottleneck').register_forward_hook(hook)
    try:
        x = torch.randn(8, 3)
        _ = m2(x)
        acts = hook.activations
        assert acts is not None
        # Dimension 0 should be identically zero
        assert torch.allclose(acts[..., 0], torch.zeros_like(acts[..., 0]))
        # Other dims should have some nonzero variance typically
        assert (acts[..., 1:].abs().sum() > 0).item()
    finally:
        handle.remove()
