from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class CameraPosition(Enum):
    FRONT = "front"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    REAR = "rear"


@dataclass
class CameraConfig:
    id: int
    position: CameraPosition
    width: int = 1280
    height: int = 720
    fps: int = 30
    enabled: bool = True


CAMERAS: Dict[str, CameraConfig] = {
    "front": CameraConfig(
        id=0,
        position=CameraPosition.FRONT,
        width=1280,
        height=720,
        fps=30,
        enabled=True
    ),
    "side_left": CameraConfig(
        id=1,
        position=CameraPosition.SIDE_LEFT,
        width=640,
        height=480,
        fps=30,
        enabled=False
    ),
    "side_right": CameraConfig(
        id=2,
        position=CameraPosition.SIDE_RIGHT,
        width=640,
        height=480,
        fps=30,
        enabled=False
    ),
}


def get_camera_config(name: str) -> Optional[CameraConfig]:
    return CAMERAS.get(name)


def get_enabled_cameras() -> Dict[str, CameraConfig]:
    return {k: v for k, v in CAMERAS.items() if v.enabled}


def enable_camera(name: str) -> bool:
    if name in CAMERAS:
        CAMERAS[name].enabled = True
        return True
    return False


def disable_camera(name: str) -> bool:
    if name in CAMERAS:
        CAMERAS[name].enabled = False
        return True
    return False
