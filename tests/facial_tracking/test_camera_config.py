import pytest

from facial_tracking.camera_config import (
    CameraPosition,
    CameraConfig,
    CAMERAS,
    get_camera_config,
    get_enabled_cameras,
    enable_camera,
    disable_camera,
)


class TestCameraPosition:
    def test_camera_position_enum(self):
        assert CameraPosition.FRONT.value == "front"
        assert CameraPosition.SIDE_LEFT.value == "side_left"


class TestCameraConfig:
    def test_default_camera_config(self):
        config = CameraConfig(id=0, position=CameraPosition.FRONT)
        assert config.id == 0
        assert config.position == CameraPosition.FRONT
        assert config.width == 1280
        assert config.height == 720
        assert config.fps == 30
        assert config.enabled is True


class TestGetCameraConfig:
    def test_get_existing_camera(self):
        config = get_camera_config("front")
        assert config is not None
        assert config.position == CameraPosition.FRONT

    def test_get_nonexistent_camera(self):
        config = get_camera_config("nonexistent")
        assert config is None


class TestGetEnabledCameras:
    def test_returns_only_enabled(self):
        cameras = get_enabled_cameras()
        assert "front" in cameras
        assert all(c.enabled for c in cameras.values())


class TestEnableDisableCamera:
    def test_enable_camera(self):
        disable_camera("front")
        assert enable_camera("front") is True

    def test_disable_camera(self):
        enable_camera("front")
        assert disable_camera("front") is True

    def test_enable_nonexistent_returns_false(self):
        assert enable_camera("nonexistent") is False

    def test_disable_nonexistent_returns_false(self):
        assert disable_camera("nonexistent") is False
