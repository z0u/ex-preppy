# DevInterp experiments with color embeddings

This is a series of experiments in which we attempt to impose structure on (latent) embeddings. Ultimately, the goal is to develop a capability to structure the latent spaces in complex models like LLMs.

## Background

Recent experiments have provided some evidence that "bad" behaviors in LLMS cluster together (such as _writing malicious code_ and _being racist_). Although surprising, it makes some intuitive sense: perhaps such behaviors cluster together because it's just the most efficient way to compress knowledge. However, _intervening_ on model behavior remains a tremendous challenge — partly because we don't know which directions in latent space correspond to undesirable traits, and we don't know how tangled up they might be with benign concepts. Indeed, attempts to align models to display "good" behavior often comes at the cost of reduced performance overall.

We hope that this research will reveal more precise and robust ways to constrain the capabilities of LLMs. In contrast to mech interp — which attempts to discover model characteristics _after_ training — we anticipate that anchoring core concepts to known directions will make alignment efforts more robust, through two mechanisms:

1. The relevant directions would be known _even before training_, so you don't need to look for them. This could improve the prospect of both measuring model alignment throughout training, and intervening on misaligned behavior after training.
2. Directions of interest should act as attractors for similar concepts, reducing the chance that unrelated (benign) concepts become entangled with them.

## 1. Preliminary experiments with color

We begin with some experiments with color, because color spaces are well defined and highly intuitive for visualization.

1. [Color data](docs/ex-1.1-color-data.ipynb): Exploration of ways to construct and visualize color spaces such as RGB and HSV.
2. [MLP bottleneck](docs/ex-1.2-color-mlp-bottleneck.ipynb): A 2-layer MLP autoencoder (extremely simple network) that squeezes bright, saturated RGB data through a 2D embedding layer. The network successfully discovers the color wheel — although it needs some help, in the form of explicit normalization.
3. [Curriculum learning](docs/ex-1.3-color-mlp-curriculum.ipynb): The same MLP, but with a 3D embedding layer. Curriculum learning and regularization are used to encourage the model to discover the color wheel without explicit normalization. The hues are embedded into the first two dimensions (as before); later phases in the curriculum add varying tones (values), which naturally balloon out into the third dimension.
4. [Parameter transitions](docs/ex-1.4-parameter-transitions.ipynb): Exploration of ways to cause hyperparameters to vary smoothly over time, both to a schedule, and in reaction to measurements during training.
5. [Smooth curriculum](docs/ex-1.5-color-mlp-anchoring.ipynb): Like experiment 1.3, but with hyperparameters smoothly varying across curriculum phases. For example, the extents of the color space of the training data (HSV) are gradually increased instead of extending it in large discrete steps.
6. [Smooth vs. stepped curricula](docs/ex-1.6-curriculum-comparison.ipynb): A direct comparison of training stability and latent space evolution when using smooth hyperparameter transitions versus traditional stepped phase changes. This experiment had a negative result: it seems the smooth transitions don't help with training dynamics (although they do make curriculum specification easier).
7. [Sparse labels for regularization](docs/ex-1.7-sparse-labels.ipynb): We do away with most of the curriculum, training on the full dataset from the start but with targeted regularization. We hope to achieve similar results to the earlier experiments, but with a more realistic training dataset: the curriculum phases were "clean" in a way that is probably hard to replicate in LLM corpora.

## Future work

- Demonstrate intervention at inference time, showing that some colors can be reliably muted without affecting those that are not "close". For example, cause the network to fail to reconstruct _red_ and colors close to red, but allow _orange_.
- Demonstrate that the latent space can be further manipulated to completely remove a representation. For example, pressure the network to reconfigure the space so that _only_ red colors are on one particular embedding dimension, and then _delete_ that dimension from the network. Hopefully, this would make it difficult to fine-tune the model later to restore the deleted capability.
- Demonstrate a proof-of-concept transformer network with similar latent space structure. It could be a very small transfomer that can perform simple color operations, such as mixing colors.

## References

This project relies on several open-source libraries.

- **Matplotlib:** Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90-95.

- **NumPy:** Harris, C.R., Millman, K.J., van der Walt, S.J. et al. (2020). Array programming with NumPy. *Nature, 585*, 357–362.

- **PyTorch:** Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In *Advances in Neural Information Processing Systems* (pp. 8026-8037).

- **scikit-image:** van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., ... & Yu, T. (2014). scikit-image: image processing in Python. *PeerJ, 2*, e453.

- **scikit-learn:** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.

- **scikit-learn API:** Buitinck, L., Louppe, G., Blondel, M., Pedregosa, F., Mueller, A., Grisel, O., ... & Varoquaux, G. (2013). API design for machine learning software: experiences from the scikit-learn project. *ECML PKDD Workshop: Languages for Data Mining and Machine Learning*, 108-122.
