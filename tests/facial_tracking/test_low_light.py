import numpy as np
import pytest

from facial_tracking.low_light import (
    LowLightConfig,
    LOW_LIGHT_CONFIG,
    detect_low_light,
    enhance_low_light,
    get_adaptive_threshold,
    apply_ir_mode,
)


class TestLowLightConfig:
    def test_default_config(self):
        config = LowLightConfig()
        assert config.enabled is False
        assert config.ir_threshold_multiplier == 1.3
        assert config.brightness_boost == 1.2
        assert config.gamma_correction == 1.5


class TestDetectLowLight:
    def test_bright_image_not_detected(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 150
        assert detect_low_light(image, threshold=80) is False

    def test_dark_image_detected(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 50
        assert detect_low_light(image, threshold=80) is True


class TestEnhanceLowLight:
    def test_disabled_returns_original(self):
        config = LowLightConfig(enabled=False)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = enhance_low_light(image, config)
        np.testing.assert_array_equal(result, image)

    def test_enabled_enhances_image(self):
        config = LowLightConfig(enabled=True, brightness_boost=1.5)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = enhance_low_light(image, config)
        assert result.shape == image.shape
        assert result.dtype == np.uint8


class TestGetAdaptiveThreshold:
    def test_disabled_returns_base(self):
        config = LowLightConfig(enabled=False)
        base = 0.22
        result = get_adaptive_threshold(base, config)
        assert result == base

    def test_enabled_multiplies_threshold(self):
        config = LowLightConfig(enabled=True, ir_threshold_multiplier=1.5)
        base = 0.22
        result = get_adaptive_threshold(base, config)
        assert result == base * 1.5


class TestApplyIrMode:
    def test_disabled_returns_original(self):
        config = LowLightConfig(enabled=False)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = apply_ir_mode(image, config)
        np.testing.assert_array_equal(result, image)

    def test_enabled_returns_same_shape(self):
        config = LowLightConfig(enabled=True)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = apply_ir_mode(image, config)
        assert result.shape == image.shape
