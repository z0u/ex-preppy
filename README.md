# DevInterp experiments with color embeddings

This is a series of experiments in which we attempt to impose structure on (latent) embeddings. Ultimately, the goal is to develop a capability to structure the latent spaces in complex models like LLMs.

<br>

## Background

Recent experiments have provided some evidence that "bad" behaviors in LLMS cluster together (such as _writing malicious code_ and _being racist_). Although surprising, it makes some intuitive sense: perhaps such behaviors cluster together because it's just the most efficient way to compress knowledge. However, _intervening_ on model behavior remains a tremendous challenge — partly because we don't know which directions in latent space correspond to undesirable traits, and we don't know how tangled up they might be with benign concepts. Indeed, attempts to align models to display "good" behavior often comes at the cost of reduced performance overall.

We hope that this research will reveal more precise and robust ways to constrain the capabilities of LLMs. In contrast to mech interp — which attempts to discover model characteristics _after_ training — we anticipate that anchoring core concepts to known directions will make alignment efforts more robust, through two mechanisms:

1. The relevant directions would be known _even before training_, so you don't need to look for them. This could improve the prospect of both measuring model alignment throughout training, and intervening on misaligned behavior after training.
2. Directions of interest should act as attractors for similar concepts, reducing the chance that unrelated (benign) concepts become entangled with them.

<br>

## M1. Preliminary experiments with color

We begin with some experiments with color, because color spaces are well defined and highly intuitive for visualization. Our goal is to demonstrate that it's possible to impose interpretable structure on latent space in a toy model.

<picture>
    <source srcset="docs/m1-color-mlp/large-assets/ex-1.7-color-phase-history.dark.png" media="(prefers-color-scheme: dark)" />
    <source srcset="docs/m1-color-mlp/large-assets/ex-1.7-color-phase-history.png" media="(prefers-color-scheme: light)" />
    <img src="docs/m1-color-mlp/large-assets/ex-1.7-color-phase-history.png" alt="Visualization of colorful latent embeddings in experiment 1.7, showing three large scatter plots from the end of training, and ten small thumbnails from earlier training steps. The thumbnails show how latent space evolves from a small cluster of dots, to a color cube, though various contortions, until finally it forms a smooth, regular sphere." />
</picture>

> Scatter plots of latent embeddings from [Ex 1.7](docs/m1-color-mlp/ex-1.7-sparse-labels.ipynb), showing 2D slices (axis-aligned projections) of the 4D activation space. Regularization encouraged hue to be represented in the first two dimensions, resulting in the model discovering the color wheel (left).

1. [Color data](docs/m1-color-mlp/ex-1.1-color-data.ipynb): Exploration of ways to construct and visualize color spaces such as RGB and HSV.
2. [MLP bottleneck](docs/m1-color-mlp/ex-1.2-color-mlp-bottleneck.ipynb): A 2-layer MLP autoencoder (extremely simple network) that squeezes bright, saturated RGB data through a 2D embedding layer. The network successfully discovers the color wheel — although it needs some help, in the form of explicit normalization.
3. [Curriculum learning](docs/m1-color-mlp/ex-1.3-color-mlp-curriculum.ipynb): The same MLP, but with a 3D embedding layer. Curriculum learning and regularization are used to encourage the model to discover the color wheel without explicit normalization. The hues are embedded into the first two dimensions (as before); later phases in the curriculum add varying tones (values), which naturally balloon out into the third dimension.
4. [Parameter transitions](docs/m1-color-mlp/ex-1.4-parameter-transitions.ipynb): Exploration of ways to cause hyperparameters to vary smoothly over time, both to a schedule, and in reaction to measurements during training.
5. [Smooth curriculum](docs/m1-color-mlp/ex-1.5-color-mlp-anchoring.ipynb): Like experiment 1.3, but with 4D embeddings, and hyperparameters smoothly varying across curriculum phases. For example, the extents of the color space of the training data (HSV) are gradually increased instead of extending it in large discrete steps.
6. [Smooth vs. stepped curricula](docs/m1-color-mlp/ex-1.6-curriculum-comparison.ipynb): A direct comparison of training stability and latent space evolution when using smooth hyperparameter transitions versus traditional stepped phase changes. This experiment had a negative result: it seems the smooth transitions don't help with training dynamics (although they do make curriculum specification easier).
7. [Sparse labels for regularization](docs/m1-color-mlp/ex-1.7-sparse-labels.ipynb): We do away with the data phases, training on the full dataset from the start but with targeted (but noisy) regularization. We achieve similar results to the earlier experiments, but with a more realistic training dataset: previously, the curriculum phases were "clean" in a way that is probably hard to replicate in LLM corpora.
8. [Regularizer combinations](docs/m1-color-mlp/ex-1.8-regularizer-combinations.ipynb): Systematic study to see the effects of each regularizer by itself, and all combinations of the regularizers. In each run, the regularizer weight schedules are kept the same, but select regularizers are not applied at all. We observe that they are all needed to produce a latent space with the desired characteristics.

<br>

MLP experiment summary:

| Ex  | Phases                   | Embeddings | Regularization terms                | Hyperparameters  |
| --- | ------------------------ | ---------- | ----------------------------------- | ---------------- |
| 1.2 | 1: Hues only             | 2D         | None (explicit normalization)       | Constant         |
| 1.3 | 5: 6 colors ~ all values | 3D         | Unitarity, planarity                | Stepped          |
| 1.5 | 4: 6 colors ~ all colors | 4D         | Unit, planar, repulsion (Euclidean) | Smooth           |
| 1.6 | 5: 6 colors ~ all colors | 4D         | Unit, planar, repulsion (Euclidean) | Smooth & stepped |
| 1.7 | 1: All colors            | 4D         | Unit, planar, repulsion (cosine)    | Smooth           |
| 1.8 | 1: All colors            | 4D         | All combinations                    | Smooth           |

<br>

Publications relating to this milestone:

> ### [Selective regularization for alignment-focused representation engineering - LessWrong](https://www.lesswrong.com/posts/HFcriD29cw3E5QLCR/selective-regularization-for-alignment-focused)
>
> We study how selective regularization during training can guide neural networks to develop predictable, interpretable latent spaces with alignment applications in mind. Using color as a test domain, we observe that anchoring even a single concept (red) influences the organization of other concepts, with related concepts clustering nearby — even with weak supervision. We then propose that concept-anchored representation engineering might enable more precise intervention in complex models without requiring extensive post-hoc interpretability work.

> ### [Side quests in curriculum learning and regularization - LessWrong](https://www.lesswrong.com/posts/TFedsvt6P68XcLK7h/side-quests-in-curriculum-learning-and-regularization)
>
> In Selective regularization for alignment-focused representation engineering, we presented a successful approach for structuring the latent space of a simple MLP. Here we document our side quests: experiments that didn't go as expected, but in which we gained experience in regularization design and training dynamics.

<br>

## M2. Practical control and intervention (IN PROGRESS)

> Okay, you can structure latent spaces... but can you actually use that structure?

In this milestone, we develop intervention functions and apply them to the structured color model from M1.

<picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/m2-control/large-assets/ex-2.1-suppression.dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/m2-control/large-assets/ex-2.1-suppression.png">
    <img alt="Plots of interventions. Top row: semicircular polar plots showing the effects of suppression on activations. Each plot shows two lobes: an orange one indicating the magnitude of the intervention, and a blue one showing the transformed activation space. The direction being intervened on (the 'subject') is always 'up', so the orange 'magnitude' lobes are also oriented upwards. The blue 'transformed' lobes are more circular but have a depression in the top, showing that the directions more aligned with the subject are squashed/attenuated by the intervention. Bottom row: line charts showing intervention strength as a function of alignment." src="docs/m2-control/large-assets/ex-2.1-suppression.png">
</picture>

> Plots of intervention functions from [Ex 2.1](docs/m2-control/ex-2.1-intervention-lobe.ipynb). **Top row:** The effects of suppression vs. alignment with a concept activation vector (up). The orange lobes indicate the magnitude of the intervention, and the blue lobes show the transformed activation space. Directions more aligned with the subject are squashed/attenuated by the intervention. **Bottom row:** line charts showing intervention strength as a function of alignment.

1. [Intervention lobes](docs/m2-control/ex-2.1-intervention-lobe.ipynb): Exploration of intervention function shape. Taking inspiration from computer graphics shader literature, we visualize intervention functions and their falloffs as polar plots. We implement two functions: suppression (which subtracts the concept vector) and repulsion (which steers activations away from the concept vector).
2. [Specific concept intervention](docs/m2-control/ex-2.2-inhibit-red.ipynb): Application of interventions to the color autoencoder. We train a bottleneck autoencoder, predict where one key concept will be located, and then intervene on its activations.
3. [Explicit normalization](docs/m2-control/ex-2.3-explicit-norm.ipynb): Improved the autoencoder model by explicitly normalizing the bottleneck activations (in addition to regularizing them to have unit norm), and by removing the sigmoid layer from the decoder. This gives a much more regular latent structure, improves reconstruction loss, and improves intervention effectiveness.
4. [Post-norm regularization](docs/m2-control/ex-2.4-post-norm-reg.ipynb): Further improved the model and intervention effectiveness by applying all regularizers except for unit norm after the explicit normalization step.
5. [Only one anchor](docs/m2-control/ex-2.5-only-red.ipynb): Demonstration of intervention without the planarity constraint. Red is still anchored at the top, but other colors are placed arbitrarily. Interventions are shown to be almost as precise.
6. [Permanent concept deletion](docs/m2-control/ex-2.6-delete-warm-cool.ipynb): Demonstrate that the latent space can be further manipulated to completely remove a concept. We train the color autoencoder such that it rediscovers the color wheel with _red_ at $(1,0,0,0)$; _cyan_ is naturally opposed to that and positions itself at $(-1,0,0,0)$. Then we modify the model parameters to delete the concept of warmth by: 1. ablation, in which the associated parameters are zeroed; 2. pruning, in which the parameters are removed (which reduces the dimensionality of the bottleneck).
7. [Subspace deletion](docs/m2-control/ex-2.7-delete-hue.ipynb): Removal of the model's ability to work with _hue_ by ablating the first two dimensions of latent space. This shows the removal of a multidimensional concept (or family of concepts, i.e. _hues_), with minimal impact on other concetps (_white_, _black_, and _grays_).
8. [Delete only red (failed)](docs/m2-control/ex-2.8-delete-only-red.ipynb): Attempt to completely remove _red_ without affecting _cyan_. We removed the planarity term and added an anti-anchor term to push colors away from being opposed to _red_. This experiment failed: ablating _red_ also heavily impacted other colors, especially desaturated ones.
9. [Delete only red](docs/m2-control/ex-2.9-delete-only-red-5d.ipynb): Completely remove _red_ without affecting _cyan_. This time we succeeded. It turns out the model needed additional capacity to warp latent space into the shape required to isolate _red_: the bottleneck needed an extra dimension, and the model needed more layers (extra nonlinearity).

- TODO: Renormalize activations after deletion.

<br>

## M3. Structured color transformer (TO DO)

Proof-of-concept transformer network with similar latent space structure. It could be a very small transfomer that can perform simple color operations, such as mixing colors.

1. Simple transformer doing color operations (mixing, complementary colors, etc.)
2. Successful transfer of anchoring techniques to the residual stream or QK space (attention mechanism), with validation that structure persists through transformer training dynamics

<br>

## M4. Language model application (TO DO)

Impose structure on the latent representations of a transformer language model.

1. Weak labeling pipeline for internet text (identifying "harmful," "deceptive," etc.)
2. Application to actual language model training
3. Evaluation of structured representations in the residual stream or QK space

<br>

---

## References

This project relies on several open-source libraries.

- **Matplotlib:** Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. _Computing in Science & Engineering, 9_(3), 90-95.

- **NumPy:** Harris, C.R., Millman, K.J., van der Walt, S.J. et al. (2020). Array programming with NumPy. _Nature, 585_, 357–362.

- **PyTorch:** Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In _Advances in Neural Information Processing Systems_ (pp. 8026-8037).

- **scikit-image:** van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., ... & Yu, T. (2014). scikit-image: image processing in Python. _PeerJ, 2_, e453.

- **scikit-learn:** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. _Journal of Machine Learning Research, 12_, 2825-2830.

- **scikit-learn API:** Buitinck, L., Louppe, G., Blondel, M., Pedregosa, F., Mueller, A., Grisel, O., ... & Varoquaux, G. (2013). API design for machine learning software: experiences from the scikit-learn project. _ECML PKDD Workshop: Languages for Data Mining and Machine Learning_, 108-122.
