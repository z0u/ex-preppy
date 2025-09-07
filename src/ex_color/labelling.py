import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate


def generate_color_labels(data: Tensor, vibrancies: Tensor) -> dict[str, Tensor]:
    """
    Generate label probabilities based on RGB values.

    Args:
        data: Batch of RGB values [B, 3]
        vibrancies: Corresponding vibrancy values for the RGB batch

    Returns:
        Dictionary mapping label names to probabilities str -> [B]
    """
    labels: dict[str, Tensor] = {}

    # Labels are assigned based on proximity to certain colors.
    # Distance is raised to a power to sharpen the association (i.e. weaken the label for colors that are futher away).

    # Proximity to primary colors
    r, g, b = data[:, 0], data[:, 1], data[:, 2]
    labels['red'] = (r * (1 - g / 2 - b / 2)) ** 10
    # labels['green'] = g * (1 - r / 2 - b / 2)
    # labels['blue'] = b * (1 - r / 2 - g / 2)

    # Proximity to any fully-saturated, fully-bright color
    labels['vibrant'] = vibrancies**100
    labels['desaturated'] = (1 - vibrancies) ** 10

    return labels


def collate_with_generated_labels(
    batch,
    *,
    soft: bool = True,
    scale: dict[str, float] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Custom collate function that generates labels for the samples.

    Args:
        batch: A list of ((data_tensor,), index_tensor) tuples from TensorDataset.
               Note: TensorDataset wraps single tensors in a tuple.
        soft: If True, return soft labels (0..1). Otherwise, return hard labels (0 or 1).
        scale: Linear scaling factors for the labels (applied before discretizing).

    Returns:
        A tuple: (collated_data_tensor, collated_labels_tensor)
    """
    # Separate data and indices
    # TensorDataset yields tuples like ((data_point_tensor,), index_scalar_tensor)
    data_tuple_list = [item[0] for item in batch]  # List of (data_tensor,) tuples
    vibrancies = [item[1] for item in batch]

    # Collate the data points using the default collate function
    # default_collate handles the list of (data_tensor,) tuples correctly
    collated_data = default_collate(data_tuple_list)
    vibrancies = default_collate(vibrancies)
    label_probs = generate_color_labels(collated_data, vibrancies)
    for k, v in (scale or {}).items():
        label_probs[k] = label_probs[k] * v

    if soft:
        # Return the probabilities directly
        return collated_data, label_probs
    else:
        # Sample labels stochastically
        labels = {k: discretize(v) for k, v in label_probs.items()}
        return collated_data, labels


def discretize(probs: Tensor) -> Tensor:
    """
    Discretize probabilities into binary labels.

    Args:
        probs: Tensor of probabilities [B]

    Returns:
        Tensor of binary labels [B]
    """
    # Sample from a uniform distribution
    rand = torch.rand_like(probs)
    return (rand < probs).float()  # Convert to float for compatibility with loss functions
