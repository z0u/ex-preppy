import torch
from torch import tensor

from ex_color.loss.anchor import AngularAnchor, AntiAnchor


def test_angular_anchor_alignment_full_sphere():
    anchor = AngularAnchor(tensor([1.0, 0.0]))
    acts = torch.stack([tensor([1.0, 0.0]), tensor([-1.0, 0.0]), tensor([0.0, 1.0])])
    loss = anchor(acts)
    # cos: 1, -1, 0 -> losses: 0, 2, 1
    assert torch.allclose(loss, tensor([0.0, 2.0, 1.0]), atol=1e-6)


def test_anti_anchor_hemi():
    anti = AntiAnchor(tensor([0.0, 1.0]), hemi=True)
    acts = torch.stack(
        [
            tensor([0.0, 1.0]),  # cos=1 -> 1
            tensor([0.0, -1.0]),  # cos=-1 -> 0
            tensor([1.0, 0.0]),  # cos=0 -> 0 (hemi clamps to ortho)
        ]
    )
    loss = anti(acts)
    assert torch.allclose(loss, tensor([1.0, 0.0, 0.0]), atol=1e-6)


def test_anti_anchor_full():
    anti = AntiAnchor(tensor([0.0, 1.0]), hemi=False)
    acts = torch.stack(
        [
            tensor([0.0, 1.0]),  # cos=1 -> (1+1)/2=1
            tensor([0.0, -1.0]),  # cos=-1 -> 0
            tensor([1.0, 0.0]),  # cos=0 -> 0.5
        ]
    )
    loss = anti(acts)
    assert torch.allclose(loss, tensor([1.0, 0.0, 0.5]), atol=1e-6)
