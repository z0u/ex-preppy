# DevInterp experiments with color embeddings

This is a series of experiments in which we attempt to impose structure on (latent) embeddings. Ultimately, the goal is to develop a capability to structure the latent spaces in complex models like LLMs.

## Background

Recent experiments have provided some evidence that "bad" behaviors in LLMS cluster together (such as _writing malicious code_ and _being racist_). Although surprising, it makes some intuitive sense: perhaps such behaviors cluster together because it's just the most efficient way to compress knowledge. However, _intervening_ on model behavior remains a tremendous challenge — partly because we don't know which directions in latent space correspond to undesirable traits, and we don't know how tangled up they might be with benign concepts. Indeed, attempts to align models to display "good" behavior often comes at the cost of reduced performance overall.

We hope that this research will reveal more precise and robust ways to constrain the capabilities of LLMs. In contrast to mech interp — which attempts to discover model characteristics _after_ training — we anticipate that anchoring core concepts to known directions will make alignment efforts more robust, through two mechanisms:

1. The relevant directions would be known _even before training_, so you don't need to look for them. This could improve the prospect of both measuring model alignment throughout training, and intervening on misaligned behavior after training.
2. Directions of interest should act as attractors for similar concepts, reducing the chance that unrelated (benign) concepts become entangled with them.

## Preliminary experiments with color

We begin with some experiments with color, because color spaces are well defined and highly intuitive for visualization.

1. [Color data](docs/ex-1.1-color-data.ipynb): Exploration of ways to construct and visualize color spaces such as RGB and HSV.
2. [MLP bottleneck](docs/ex-1.2-color-mlp-bottleneck.ipynb): A 2-layer MLP autoencoder (extremely simple network) that squeezes bright, saturated RGB data through a 2D embedding layer. The network successfully discovers the color wheel — although it needs some help, in the form of explicit normalization.
3. [Curriculum learning](docs/ex-1.3-color-mlp-curriculum.ipynb): The same MLP, but with a 3D embedding layer. Curriculum learning and regularization are used to encourage the model to discover the color wheel without explicit normalization. The hues are embedded into the first two dimensions (as before); later phases in the curriculum add varying tones (values), which naturally balloon out into the third dimension.
4. [Parameter transitions](docs/ex-1.4-parameter-transitions.ipynb): Exploration of ways to cause hyperparameters to vary smoothly over time, both to a schedule, and in reaction to measurements during training.
5. [Smooth curriculum](docs/ex-1.5-color-mlp-anchoring.ipynb): Like experiment 1.3, but with hyperparameters smoothly varying across curriculum phases. For example, the extents of the color space of the training data (HSV) are gradually increased instead of extending it in large discrete steps.

## Future work

- Demonstrate intervention at inference time, showing that some colors can be reliably muted without affecting those that are not "close". For example, cause the network to fail to reconstruct _red_ and colors close to red, but allow _orange_.
- Demonstrate that the latent space can be further manipulated to completely remove a representation. For example, pressure the network to reconfigure the space so that _only_ red colors are on one particular embedding dimension, and then _delete_ that dimension from the network. Hopefully, this would make it difficult to fine-tune the model later to restore the deleted capability.
- Demonstrate a proof-of-concept transformer network with similar latent space structure. It could be a very small transfomer that can perform simple color operations, such as mixing colors.
