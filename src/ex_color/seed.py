import logging

import lightning as L
import torch

log = logging.getLogger(__name__)


def set_deterministic_mode(seed: int):
    """Make experiments reproducible."""
    L.seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    log.info('PyTorch set to deterministic mode')
