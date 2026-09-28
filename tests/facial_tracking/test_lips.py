import numpy as np
import pytest
from unittest.mock import MagicMock

import facial_tracking.conf as conf
from facial_tracking.lips import Lips


class TestLipsOpen:
    def test_mouth_open_detected(self, mock_mouth_open_frame, lips_landmarks):
        frame, face_landmarks = mock_mouth_open_frame
        lips = Lips(frame, face_landmarks, lips_landmarks)
        assert lips.mouth_open() is True

    def test_mouth_closed_detected(self, mock_mouth_closed_frame, lips_landmarks):
        frame, face_landmarks = mock_mouth_closed_frame
        lips = Lips(frame, face_landmarks, lips_landmarks)
        assert lips.mouth_open() is False

    def test_mouth_open_ratio_above_threshold(self, mock_mouth_open_frame, lips_landmarks):
        frame, face_landmarks = mock_mouth_open_frame
        lips = Lips(frame, face_landmarks, lips_landmarks)
        assert lips.mouth_open_ratio > conf.MOUTH_OPEN

    def test_mouth_open_ratio_below_threshold(self, mock_mouth_closed_frame, lips_landmarks):
        frame, face_landmarks = mock_mouth_closed_frame
        lips = Lips(frame, face_landmarks, lips_landmarks)
        assert lips.mouth_open_ratio < conf.MOUTH_OPEN


class TestLipsPosition:
    def test_lips_position_extraction(self, mock_mouth_open_frame, lips_landmarks):
        frame, face_landmarks = mock_mouth_open_frame
        lips = Lips(frame, face_landmarks, lips_landmarks)
        pos = lips.pos
        assert len(pos) == 4
        for p in pos:
            assert len(p) == 2
            assert all(isinstance(c, int) for c in p)


class TestLipsDraw:
    def test_draw_lips_does_not_raise(self, mock_mouth_open_frame, lips_landmarks):
        frame, face_landmarks = mock_mouth_open_frame
        lips = Lips(frame, face_landmarks, lips_landmarks)
        lips.draw_lips()


class TestLipsInitialization:
    def test_lips_initializes_with_pos(self, mock_mouth_open_frame, lips_landmarks):
        frame, face_landmarks = mock_mouth_open_frame
        lips = Lips(frame, face_landmarks, lips_landmarks)
        assert lips.pos is not None
        assert len(lips.pos) == 4

    def test_lips_initializes_with_mouth_open_ratio(self, mock_mouth_open_frame, lips_landmarks):
        frame, face_landmarks = mock_mouth_open_frame
        lips = Lips(frame, face_landmarks, lips_landmarks)
        assert lips.mouth_open_ratio is not None
        assert isinstance(lips.mouth_open_ratio, (int, float))


class TestLipsCustomThreshold:
    def test_mouth_open_with_custom_threshold(self, mock_mouth_open_frame, lips_landmarks):
        frame, face_landmarks = mock_mouth_open_frame
        lips = Lips(frame, face_landmarks, lips_landmarks)
        assert lips.mouth_open(threshold=10.0) is False
        assert lips.mouth_open(threshold=0.1) is True

    def test_mouth_closed_with_custom_threshold(self, mock_mouth_closed_frame, lips_landmarks):
        frame, face_landmarks = mock_mouth_closed_frame
        lips = Lips(frame, face_landmarks, lips_landmarks)
        assert lips.mouth_open(threshold=0.01) is True


class TestYawnDetection:
    def test_yawn_ratio_calculation(self, mock_mouth_open_frame, lips_landmarks):
        frame, face_landmarks = mock_mouth_open_frame
        lips = Lips(frame, face_landmarks, lips_landmarks)
        h, w = frame.shape[:2]
        expected_ratio = (lips.pos[3][1] - lips.pos[2][1]) / (lips.pos[0][0] - lips.pos[1][0])
        assert abs(lips.mouth_open_ratio - expected_ratio) < 0.001
