from typing import TypeVar
import time

from torch import nn
import modal
import io
import torch

M = TypeVar('M', bound=nn.Module)


def load_checkpoint_from_volume(model: M, volume: modal.Volume, key: str, *, retries: int = 5) -> M:
    for _ in range(3):  # sometimes volume isn't immediately ready
        try:
            buf = io.BytesIO()
            volume.read_file_into_fileobj(key, buf)
            buf.seek(0)
            state_dict = torch.load(buf, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except FileNotFoundError:
            time.sleep(2)
    raise FileNotFoundError(f'Failed to load checkpoint {key} after {retries} attempts')
