import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import facial_tracking.conf as conf
from facial_tracking.eye import Eye


class TestEyeGaze:
    def test_gaze_left_detected(self, mock_gaze_left_frame, eye_landmarks):
        frame, face_landmarks = mock_gaze_left_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        assert eye.gaze_left() is True
        assert eye.gaze_right() is False

    def test_gaze_right_detected(self, mock_gaze_right_frame, eye_landmarks):
        frame, face_landmarks = mock_gaze_right_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        assert eye.gaze_right() is True
        assert eye.gaze_left() is False

    def test_gaze_center_when_within_threshold(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        assert eye.gaze_center() is True

    def test_gaze_ratio_values(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        ratios = eye.iris_relative_to_eye
        assert len(ratios) == 3
        assert all(isinstance(r, (int, float)) for r in ratios)


class TestEyeClosed:
    def test_eye_closed_when_ratio_below_threshold(self, mock_eye_closed_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_closed_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        assert eye.eye_closed() is True

    def test_eye_open_when_ratio_above_threshold(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        assert eye.eye_closed() is False

    def test_blink_ratio_calculation(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        ratio = eye.eye_veti_to_hori
        assert isinstance(ratio, (int, float))
        assert ratio >= 0


class TestEyePosition:
    def test_eye_position_extraction(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        pos = eye.pos
        assert len(pos) == 4
        for p in pos:
            assert len(p) == 2
            assert all(isinstance(c, int) for c in p)


class TestEyeDraw:
    def test_draw_eye_does_not_raise(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        eye.draw_eye()


class TestEyeInitialization:
    def test_eye_initializes_with_iris(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        assert eye.iris is not None

    def test_eye_initializes_with_pos(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        assert eye.pos is not None
        assert len(eye.pos) == 4

    def test_eye_initializes_with_iris_relative_to_eye(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        assert eye.iris_relative_to_eye is not None
        assert len(eye.iris_relative_to_eye) == 3

    def test_eye_initializes_with_eye_veti_to_hori(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        assert eye.eye_veti_to_hori is not None


class TestEyeCustomThreshold:
    def test_gaze_left_with_custom_threshold(self, mock_gaze_left_frame, eye_landmarks):
        frame, face_landmarks = mock_gaze_left_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        assert eye.gaze_left(threshold=0.1) is True
        assert eye.gaze_left(threshold=0.5) is False

    def test_gaze_right_with_custom_threshold(self, mock_gaze_right_frame, eye_landmarks):
        frame, face_landmarks = mock_gaze_right_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        assert eye.gaze_right(threshold=0.9) is True
        assert eye.gaze_right(threshold=0.5) is False

    def test_eye_closed_with_custom_threshold(self, mock_eye_closed_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_closed_frame
        eye = Eye(frame, face_landmarks, eye_landmarks["left"])
        assert eye.eye_closed(threshold=0.5) is True
        assert eye.eye_closed(threshold=0.1) is False
