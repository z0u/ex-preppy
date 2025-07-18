import itertools
import logging
from functools import partial

import numpy as np
import torch
from torch._tensor import Tensor
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ex_color.regularizers.anchor import Anchor
from ex_color.regularizers.regularizer import RegularizerConfig
from ex_color.regularizers.planarity import planarity
from ex_color.regularizers.separate import Separate
from ex_color.regularizers.unitarity import unitarity
from ex_color.data.color_cube import ColorCube
from ex_color.data.cube_sampler import vibrancy
from ex_color.data.cyclic import arange_cyclic
from ex_color.labelling import collate_with_generated_labels
from ex_color.model import ColorMLPTrainingModule
from ex_color.seed import set_deterministic_mode
from ex_color.train_lightning import train_color_model_lightning
from mini.temporal.dopesheet import Dopesheet
from utils.logging import SimpleLoggingConfig

log = logging.getLogger(__name__)


def load_dopesheet():
    return Dopesheet.from_csv('docs/machinery-dopesheet.csv')


def prep_data() -> tuple[DataLoader, Tensor]:
    """
    Prepare data for training.

    Returns: (train, val)
    """
    hsv_cube = ColorCube.from_hsv(
        h=arange_cyclic(step_size=10 / 360),
        s=np.linspace(0, 1, 10),
        v=np.linspace(0, 1, 10),
    )
    hsv_tensor = torch.tensor(hsv_cube.rgb_grid.reshape(-1, 3), dtype=torch.float32)
    vibrancy_tensor = torch.tensor(vibrancy(hsv_cube).flatten(), dtype=torch.float32)
    hsv_dataset = TensorDataset(hsv_tensor, vibrancy_tensor)

    labeller = partial(
        collate_with_generated_labels,
        soft=False,  # Use binary labels (stochastic) to simulate the labelling of internet text
        scale={'red': 0.5, 'vibrant': 0.5},
    )
    # Desaturated and dark colors are over-represented in the cube, so we use a weighted sampler to balance them out
    hsv_loader = DataLoader(
        hsv_dataset,
        batch_size=64,
        sampler=WeightedRandomSampler(
            weights=hsv_cube.bias.flatten().tolist(),
            num_samples=len(hsv_dataset),
            replacement=True,
        ),
        collate_fn=labeller,
    )

    rgb_cube = ColorCube.from_rgb(
        r=np.linspace(0, 1, 8),
        g=np.linspace(0, 1, 8),
        b=np.linspace(0, 1, 8),
    )
    rgb_tensor = torch.tensor(rgb_cube.rgb_grid.reshape(-1, 3), dtype=torch.float32)
    return hsv_loader, rgb_tensor


ALL_REGULARIZERS = [
    RegularizerConfig(
        name='reg-polar',
        regularizer=Anchor(torch.tensor([1, 0, 0, 0], dtype=torch.float32)),
        label_affinities={'red': 1.0},
    ),
    RegularizerConfig(
        name='reg-separate',
        regularizer=Separate(power=10.0, shift=False),
        label_affinities=None,
    ),
    RegularizerConfig(
        name='reg-planar',
        regularizer=planarity,
        label_affinities={'vibrant': 1.0},
    ),
    RegularizerConfig(
        name='reg-norm-v',
        regularizer=unitarity,
        label_affinities={'vibrant': 1.0},
    ),
    RegularizerConfig(
        name='reg-norm',
        regularizer=unitarity,
        label_affinities=None,
    ),
]


def train(
    dopesheet: Dopesheet,
    regularizers: list[RegularizerConfig],
):
    """Train the model with the given dopesheet and variant."""
    log.info(f'Training with: {[r.name for r in regularizers]}')

    seed = 0
    set_deterministic_mode(seed)

    hsv_loader, rgb_tensor = prep_data()

    # Create a temporary training module just to count parameters
    temp_module = ColorMLPTrainingModule(dopesheet, torch.nn.MSELoss(), regularizers)
    total_params = sum(p.numel() for p in temp_module.parameters() if p.requires_grad)
    log.debug(f'Model initialized with {total_params:,} trainable parameters.')

    metrics_callback = train_color_model_lightning(
        hsv_loader,
        rgb_tensor,
        dopesheet,
        torch.nn.MSELoss(),
        regularizers,
    )

    return metrics_callback.history


def main():
    all_regs = ALL_REGULARIZERS
    all_combinations = list(
        itertools.chain(*(itertools.combinations(all_regs, i) for i in range(1, len(all_regs) + 1)))
    )

    combinations = all_combinations[-1:]  # For testing, select a subset
    log.info(f'Running {len(combinations):d}/{len(all_combinations):d} combinations of {len(all_regs)} regularizers.')

    runs: dict[str, list] = {}
    for combo in combinations:
        dopesheet = load_dopesheet()
        combo_list = list(combo)
        combo_name = ' + '.join(r.name for r in combo_list)
        runs[combo_name] = train(dopesheet, combo_list)
    print(runs)


if __name__ == '__main__':
    SimpleLoggingConfig().info('__main__', 'ex_color', 'utils').apply()
    main()
