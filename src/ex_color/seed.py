import logging
import random

import numpy as np
import torch

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info(f'Global random seed set to {seed}')


def set_deterministic_mode(seed: int):
    """Make experiments reproducible."""
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info('PyTorch set to deterministic mode')
