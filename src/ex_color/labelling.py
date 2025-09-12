import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate


def red_labels(data: Tensor):
    r, g, b = data[:, 0], data[:, 1], data[:, 2]
    return r * (1 - g / 2 - b / 2)


def green_labels(data: Tensor):
    r, g, b = data[:, 0], data[:, 1], data[:, 2]
    return g * (1 - r / 2 - b / 2)


def blue_labels(data: Tensor):
    r, g, b = data[:, 0], data[:, 1], data[:, 2]
    return b * (1 - r / 2 - g / 2)


def collate_with_generated_labels(
    batch,
    *,
    soft: bool = True,
    red: float = 0.0,
    green: float = 0.0,
    blue: float = 0.0,
    vibrant: float = 0.0,
    desaturated: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Custom collate function that generates labels for the samples.

    Args:
        batch: A list of ((data_tensor,), index_tensor) tuples from TensorDataset.
               Note: TensorDataset wraps single tensors in a tuple.
        soft: If True, return soft labels (0..1). Otherwise, return hard labels (0 or 1).
        red: Linear scaling factor for the `red` label (applied before discretizing).
        green: Linear scaling factor for the `green` label (applied before discretizing).
        blue: Linear scaling factor for the `blue` label (applied before discretizing).
        vibrant: Linear scaling factor for the `vibrant` label (applied before discretizing).
        desaturated: Linear scaling factor for the `desaturated` label (applied before discretizing).

    Returns:
        A tuple: (collated_data_tensor, collated_labels_tensor)
    """
    # Separate data and indices
    # TensorDataset yields tuples like ((data_point_tensor,), index_scalar_tensor)
    color_samples = [item[0] for item in batch]  # List of (data_tensor,) tuples
    vibrancy_samples = [item[1] for item in batch]

    # Collate the data points using the default collate function
    # default_collate handles the list of (data_tensor,) tuples correctly
    colors: Tensor = default_collate(color_samples)
    vibrancies: Tensor = default_collate(vibrancy_samples)

    label_probs: dict[str, Tensor] = {}
    if red:
        label_probs['red'] = red_labels(colors) ** 10 * red
    if green:
        label_probs['green'] = green_labels(colors) ** 10 * green
    if blue:
        label_probs['blue'] = blue_labels(colors) ** 10 * blue
    if vibrant:
        label_probs['vibrant'] = vibrancies**100 * vibrant
    if desaturated:
        label_probs['desaturated'] = (1 - vibrancies) ** 10 * desaturated

    if soft:
        # Return the probabilities directly
        return colors, label_probs
    else:
        # Sample labels stochastically
        labels = {k: discretize(v) for k, v in label_probs.items()}
        return colors, labels


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
