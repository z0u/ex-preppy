# This file is used to allowlist unused code in the project.
# https://github.com/jendrikseipp/vulture?tab=readme-ov-file#handling-false-positives
# type: ignore


exception  # unused variable (src/mini/_state.py:22)
_.exception  # unused attribute (src/mini/experiment.py:282)
_.before_each  # unused method (src/mini/experiment.py:170)
_.after_each  # unused method (src/mini/experiment.py:170)
_.hither  # unused attribute (src/mini/experiment.py:94)

_.base_level  # unused method (src/utils/logging.py:66)
_.to_stream  # unused method (src/utils/logging.py:71)
_.critical  # unused method (src/utils/logging.py:76)
_.trace  # unused method (src/utils/logging.py:101)
_.format  # unused method (src/utils/logging.py:26)

_._repr_html_  # Jupyter renderer
owner  # Descriptor protocol
__call__  # Function protocol

bottom  # unused variable (src/utils/theming.py:290)
left  # unused variable (src/utils/theming.py:288)
right  # unused variable (src/utils/theming.py:289)
top  # unused variable (src/utils/theming.py:287)
id_sequence  # unused variable (src/utils/dom.py:27)
total_steps  # unused variable (src/utils/lr_finder/types.py:42)
lr_finder_search  # unused function (src/utils/lr_finder/lr_finder.py:17)
svg_theme_toggle  # unused function (src/utils/theming.py:122)

EntropySeries  # unused class (src/subline/series.py:26)

author  # unused variable (src/experiment/config.py:57)
fixes  # unused variable (src/experiment/config.py:62)
language  # unused variable (src/experiment/config.py:68)
total_chars  # unused variable (src/experiment/config.py:65)
total_chars  # unused variable (src/experiment/config.py:79)
total_tokens  # unused variable (src/experiment/config.py:76)
training_tokens  # unused variable (src/experiment/training/metrics.py:8)
val_loss  # unused variable (src/experiment/training/metrics.py:7)

lime  # unused variable (src/ex_color/data.py:120)
azure  # unused variable (src/ex_color/data.py:125)
pink  # unused variable (src/ex_color/data.py:129)
phases  # unused variable (.vulture-cache/ex-1.4-color-mlp-anchoring.py:292)
_.forward  # unused method (from models defined in notebooks)

mock_connect # unused variable, test fixture
_.side_effect # Mock

torch.backends.cudnn.deterministic  # torch setting
torch.backends.cudnn.benchmark  # torch setting
