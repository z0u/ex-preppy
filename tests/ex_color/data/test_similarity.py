import numpy as np
import pytest
from pytest import approx

from ex_color.data.color import hues12, grays5
from ex_color.data.similarity import hsv_similarity


@pytest.fixture
def named_colors():
    """Convert named colors to HSV arrays."""
    colors = {}
    for name, hue in hues12.items():
        colors[name] = np.array([hue, 1.0, 1.0])
    for name, value in grays5.items():
        colors[name] = np.array([0.0, 0.0, value])
    return colors


class TestHsvSimilaritySelfComparison:
    """Test that colors are maximally similar to themselves."""

    @pytest.mark.parametrize('mode', ['cosine', 'angular'])
    @pytest.mark.parametrize('hemi', [True, False])
    def test_identical_colors_perfect_similarity(self, named_colors, mode, hemi):
        """Identical colors should have similarity of 1.0."""
        red = named_colors['red']
        similarity = hsv_similarity(red, red, mode=mode, hemi=hemi)
        assert similarity == approx(1.0)

    @pytest.mark.parametrize('mode', ['cosine', 'angular'])
    @pytest.mark.parametrize('hemi', [True, False])
    def test_all_named_colors_self_similar(self, named_colors, mode, hemi):
        """All named colors should be maximally similar to themselves."""
        for name, color in named_colors.items():
            similarity = hsv_similarity(color, color, mode=mode, hemi=hemi)
            assert similarity == approx(1.0), f'{name} should have self-similarity of 1.0'


class TestHsvSimilarityOppositeColors:
    """Test similarity between opposite hues on the color wheel."""

    @pytest.mark.parametrize('mode', ['cosine', 'angular'])
    def test_complementary_colors_cosine_hemi(self, named_colors, mode):
        """Complementary colors (180° apart) should have similarity of 0 in hemi mode."""
        red = named_colors['red']
        cyan = named_colors['cyan']
        similarity = hsv_similarity(red, cyan, mode=mode, hemi=True)
        assert similarity == approx(0.0)

    @pytest.mark.parametrize('mode', ['cosine', 'angular'])
    def test_complementary_colors_full_range(self, named_colors, mode):
        """Complementary colors (180° apart) have 0 similarity even in full range mode.

        In full range mode, the hue similarity component is 0 at 180°:
        - cosine: 0.5 + cos(180°)/2 = 0.5 + (-1)/2 = 0
        - angular: (180 - 180)/180 = 0
        Since similarity is a product, the result is 0.
        """
        red = named_colors['red']
        cyan = named_colors['cyan']
        similarity = hsv_similarity(red, cyan, mode=mode, hemi=False)
        assert similarity == approx(0.0)


class TestHsvSimilarityAdjacentColors:
    """Test similarity between adjacent colors on the color wheel."""

    def test_adjacent_primaries_and_secondaries(self, named_colors):
        """Adjacent colors should have high similarity."""
        # Red (0°) and orange (30°)
        red = named_colors['red']
        orange = named_colors['orange']
        similarity = hsv_similarity(red, orange, mode='cosine', hemi=False)
        assert similarity > 0.9

    def test_adjacent_colors_green_cyan(self, named_colors):
        """Green (120°) and cyan (180°) are 60° apart."""
        green = named_colors['green']
        cyan = named_colors['cyan']
        similarity = hsv_similarity(green, cyan, mode='cosine', hemi=False)
        assert 0.5 < similarity < 0.9


class TestHsvSimilaritySaturationValue:
    """Test how saturation and value differences affect similarity."""

    def test_saturation_reduces_similarity(self):
        """Reducing saturation should reduce similarity."""
        red_full = np.array([0.0, 1.0, 1.0])
        red_half = np.array([0.0, 0.5, 1.0])
        similarity = hsv_similarity(red_full, red_half, mode='cosine', hemi=False)
        assert 0.4 < similarity < 0.6  # Should be around 0.5 due to 50% sat difference

    def test_value_reduces_similarity(self):
        """Reducing value should reduce similarity."""
        red_full = np.array([0.0, 1.0, 1.0])
        red_half = np.array([0.0, 1.0, 0.5])
        similarity = hsv_similarity(red_full, red_half, mode='cosine', hemi=False)
        assert 0.4 < similarity < 0.6  # Should be around 0.5 due to 50% value difference

    def test_both_sat_value_reduce_similarity(self):
        """Reducing both saturation and value compounds the effect."""
        red_full = np.array([0.0, 1.0, 1.0])
        red_half = np.array([0.0, 0.5, 0.5])
        similarity = hsv_similarity(red_full, red_half, mode='cosine', hemi=False)
        assert 0.2 < similarity < 0.3  # Should be around 0.25 (0.5 * 0.5)


class TestHsvSimilarityGrays:
    """Test similarity between grayscale colors."""

    @pytest.mark.parametrize('mode', ['cosine', 'angular'])
    @pytest.mark.parametrize('hemi', [True, False])
    def test_black_white_similarity(self, named_colors, mode, hemi):
        """Black and white differ only in value."""
        black = named_colors['black']
        white = named_colors['white']
        similarity = hsv_similarity(black, white, mode=mode, hemi=hemi)
        # They differ by 1.0 in value, so similarity should be 0
        assert similarity == approx(0.0)

    @pytest.mark.parametrize('mode', ['cosine', 'angular'])
    @pytest.mark.parametrize('hemi', [True, False])
    def test_adjacent_grays(self, named_colors, mode, hemi):
        """Adjacent grays should have moderate similarity."""
        black = named_colors['black']
        dark_gray = named_colors['dark gray']
        similarity = hsv_similarity(black, dark_gray, mode=mode, hemi=hemi)
        # They differ by 0.25 in value, so similarity should be 0.75
        assert similarity == approx(0.75)


class TestHsvSimilarityModes:
    """Test differences between cosine and angular modes."""

    def test_cosine_vs_angular_90_degrees(self):
        """At 90 degrees, cosine mode should give 0.5, angular should give 0.5."""
        red = np.array([0.0, 1.0, 1.0])
        yellow = np.array([60 / 360, 1.0, 1.0])  # 60 degrees apart

        cosine_sim = hsv_similarity(red, yellow, mode='cosine', hemi=False)
        angular_sim = hsv_similarity(red, yellow, mode='angular', hemi=False)

        # Both should be greater than 0.5 for 60° separation
        assert cosine_sim > 0.5
        assert angular_sim > 0.5

    def test_hemi_mode_negative_cosine(self):
        """In hemi mode, negative cosine values should be clipped to 0."""
        red = np.array([0.0, 1.0, 1.0])
        cyan = np.array([180 / 360, 1.0, 1.0])  # 180 degrees (opposite)

        similarity = hsv_similarity(red, cyan, mode='cosine', hemi=True)
        assert similarity == approx(0.0)


class TestHsvSimilarityBatched:
    """Test that the function works with batched inputs."""

    def test_batched_comparison(self, named_colors):
        """Test comparing multiple colors at once."""
        # Create a batch of colors
        colors_batch = np.array(
            [
                named_colors['red'],
                named_colors['green'],
                named_colors['blue'],
            ]
        )

        # Compare to red
        red = named_colors['red']
        similarities = hsv_similarity(colors_batch, red, mode='cosine', hemi=False)

        assert similarities.shape == (3,)
        assert similarities[0] == approx(1.0)  # Red to red
        assert similarities[1] < similarities[0]  # Green to red
        assert similarities[2] < similarities[0]  # Blue to red

    def test_broadcasting_shapes(self):
        """Test that broadcasting works correctly."""
        # [2, 3] vs [3]
        colors1 = np.array(
            [
                [0.0, 1.0, 1.0],
                [0.5, 1.0, 1.0],
            ]
        )
        color2 = np.array([0.0, 1.0, 1.0])

        similarities = hsv_similarity(colors1, color2, mode='cosine', hemi=False)
        assert similarities.shape == (2,)
        assert similarities[0] == approx(1.0)
        assert similarities[1] < 1.0


class TestHsvSimilarityEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize('mode', ['cosine', 'angular'])
    @pytest.mark.parametrize('hemi', [True, False])
    def test_zero_saturation_colors(self, mode, hemi):
        """Colors with zero saturation should ignore hue differences.

        Since hue is undefined for grays (zero saturation), the function weights
        hue similarity by vibrancy (s*v). When both colors have zero saturation,
        vibrancy is 0, so hue_similarity is ignored and defaults to 1.0.
        Two grays with the same value should be perfectly similar regardless of hue.
        """
        gray1 = np.array([0.0, 0.0, 0.5])
        gray2 = np.array([0.5, 0.0, 0.5])  # Different hue but same gray

        similarity = hsv_similarity(gray1, gray2, mode=mode, hemi=hemi)
        # Hue doesn't matter for grays - they should be perfectly similar
        assert similarity == approx(1.0)

    def test_wrap_around_hue(self):
        """Test that hue wraps around correctly (359° vs 1°)."""
        color1 = np.array([1 / 360, 1.0, 1.0])
        color2 = np.array([359 / 360, 1.0, 1.0])

        similarity = hsv_similarity(color1, color2, mode='cosine', hemi=False)
        # These are only 2° apart, should be very similar
        assert similarity > 0.99
