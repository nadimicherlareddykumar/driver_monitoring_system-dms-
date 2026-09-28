import cv2
import numpy as np
from typing import Tuple, Optional


class DataAugmenter:
    def __init__(self, brightness_range=0.2, rotation_range=15, noise_std=0.01, crop_factor=0.1):
        self.brightness_range = brightness_range
        self.rotation_range = rotation_range
        self.noise_std = noise_std
        self.crop_factor = crop_factor

    def random_brightness(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        if factor is None:
            factor = 1.0 + np.random.uniform(-self.brightness_range, self.brightness_range)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def random_rotation(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        if angle is None:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    def random_noise(self, image: np.ndarray, std: Optional[float] = None) -> np.ndarray:
        if std is None:
            std = self.noise_std
        noise = np.random.normal(0, std * 255, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def random_crop_and_resize(self, image: np.ndarray, crop_factor: Optional[float] = None) -> np.ndarray:
        if crop_factor is None:
            crop_factor = self.crop_factor
        h, w = image.shape[:2]
        new_h = int(h * (1 - crop_factor))
        new_w = int(w * (1 - crop_factor))
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        cropped = image[top:top + new_h, left:left + new_w]
        return cv2.resize(cropped, (w, h))

    def random_horizontal_flip(self, image: np.ndarray, p: float = 0.5) -> np.ndarray:
        if np.random.random() < p:
            return cv2.flip(image, 1)
        return image

    def random_gamma_correction(self, image: np.ndarray, gamma_range: Tuple[float, float] = (0.7, 1.5)) -> np.ndarray:
        gamma = np.random.uniform(*gamma_range)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)

    def augment(self, image: np.ndarray, augmentations: Optional[list] = None) -> np.ndarray:
        if augmentations is None:
            augmentations = ['brightness', 'rotation', 'noise', 'flip']
        result = image.copy()
        for aug in augmentations:
            if aug == 'brightness':
                result = self.random_brightness(result)
            elif aug == 'rotation':
                result = self.random_rotation(result)
            elif aug == 'noise':
                result = self.random_noise(result)
            elif aug == 'flip':
                result = self.random_horizontal_flip(result)
            elif aug == 'gamma':
                result = self.random_gamma_correction(result)
            elif aug == 'crop':
                result = self.random_crop_and_resize(result)
        return result
