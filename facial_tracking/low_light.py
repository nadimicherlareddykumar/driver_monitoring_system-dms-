import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LowLightConfig:
    enabled: bool = False
    ir_threshold_multiplier: float = 1.3
    brightness_boost: float = 1.2
    gamma_correction: float = 1.5
    adaptive_threshold_offset: float = 0.05


LOW_LIGHT_CONFIG = LowLightConfig()


def detect_low_light(frame: np.ndarray, threshold: int = 80) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < threshold


def enhance_low_light(frame: np.ndarray, config: Optional[LowLightConfig] = None) -> np.ndarray:
    if config is None:
        config = LOW_LIGHT_CONFIG

    if not config.enabled:
        return frame

    enhanced = frame.copy()

    enhanced = cv2.convertScaleAbs(enhanced, alpha=config.brightness_boost, beta=0)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    enhanced = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    if config.gamma_correction != 1.0:
        inv_gamma = 1.0 / config.gamma_correction
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, table)

    return enhanced


def get_adaptive_threshold(base_threshold: float, config: Optional[LowLightConfig] = None) -> float:
    if config is None:
        config = LOW_LIGHT_CONFIG

    if config.enabled:
        return base_threshold * config.ir_threshold_multiplier
    return base_threshold


def apply_ir_mode(frame: np.ndarray, config: Optional[LowLightConfig] = None) -> np.ndarray:
    if config is None:
        config = LOW_LIGHT_CONFIG

    if not config.enabled:
        return frame

    enhanced = enhance_low_light(frame, config)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, ir_binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    ir_mask = cv2.cvtColor(ir_binary, cv2.COLOR_GRAY2BGR) / 255.0

    result = enhanced * ir_mask + frame * (1 - ir_mask)
    return result.astype(np.uint8)
