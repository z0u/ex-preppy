from torch import Tensor


class Falloff:
    """
    Determine intervention amount based on distance to subject.

    Args:
        alignment: Closeness of activations from the subject, of shape [B],
        where 1 is "perfectly aligned" and 0 is "not at all aligned". This could
        be cosine distance or some other measure.

    Returns:
        The offset from the original distance, of shape [B].
    """

    def __call__(self, alignment: Tensor, /) -> Tensor: ...
