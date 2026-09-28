import numpy as np
import pytest
import cv2
from unittest.mock import patch

from dms_utils.data_augmentation import DataAugmenter


class TestDataAugmenterInit:
    def test_default_initialization(self):
        augmenter = DataAugmenter()
        assert augmenter.brightness_range == 0.2
        assert augmenter.rotation_range == 15
        assert augmenter.noise_std == 0.01
        assert augmenter.crop_factor == 0.1

    def test_custom_initialization(self):
        augmenter = DataAugmenter(brightness_range=0.3, rotation_range=20)
        assert augmenter.brightness_range == 0.3
        assert augmenter.rotation_range == 20


class TestRandomBrightness:
    def test_brightness_factor_applied(self):
        augmenter = DataAugmenter()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmenter.random_brightness(image, factor=1.5)
        assert result.shape == image.shape

    def test_brightness_preserves_dtype(self):
        augmenter = DataAugmenter()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmenter.random_brightness(image, factor=1.2)
        assert result.dtype == np.uint8


class TestRandomRotation:
    def test_rotation_preserves_shape(self):
        augmenter = DataAugmenter()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmenter.random_rotation(image, angle=45)
        assert result.shape == image.shape

    def test_rotation_with_zero_angle(self):
        augmenter = DataAugmenter()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmenter.random_rotation(image, angle=0)
        np.testing.assert_array_equal(result, image)


class TestRandomNoise:
    def test_noise_preserves_shape(self):
        augmenter = DataAugmenter()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmenter.random_noise(image)
        assert result.shape == image.shape

    def test_noise_preserves_dtype(self):
        augmenter = DataAugmenter()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmenter.random_noise(image)
        assert result.dtype == np.uint8


class TestRandomCropAndResize:
    def test_crop_and_resize_preserves_shape(self):
        augmenter = DataAugmenter()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmenter.random_crop_and_resize(image)
        assert result.shape == image.shape


class TestRandomHorizontalFlip:
    def test_flip_with_probability_one(self):
        augmenter = DataAugmenter()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmenter.random_horizontal_flip(image, p=1.0)
        expected = cv2.flip(image, 1)
        np.testing.assert_array_equal(result, expected)

    def test_flip_with_probability_zero(self):
        augmenter = DataAugmenter()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmenter.random_horizontal_flip(image, p=0.0)
        np.testing.assert_array_equal(result, image)


class TestRandomGammaCorrection:
    def test_gamma_correction_preserves_shape(self):
        augmenter = DataAugmenter()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmenter.random_gamma_correction(image)
        assert result.shape == image.shape


class TestAugment:
    def test_augment_with_default_augmentations(self):
        augmenter = DataAugmenter()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmenter.augment(image)
        assert result.shape == image.shape

    def test_augment_with_empty_list(self):
        augmenter = DataAugmenter()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = augmenter.augment(image, augmentations=[])
        np.testing.assert_array_equal(result, image)
