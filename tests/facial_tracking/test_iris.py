import numpy as np
import pytest
from unittest.mock import MagicMock

from facial_tracking.iris import Iris


class TestIrisPosition:
    def test_iris_position_extraction(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        iris = Iris(frame, face_landmarks, eye_landmarks["left"])
        pos = iris.pos
        assert len(pos) == 5
        for p in pos:
            assert len(p) == 2
            assert all(isinstance(c, int) for c in p)


class TestIrisDraw:
    def test_draw_iris_does_not_raise(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        iris = Iris(frame, face_landmarks, eye_landmarks["left"])
        iris.draw_iris()

    def test_draw_iris_with_border_does_not_raise(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        iris = Iris(frame, face_landmarks, eye_landmarks["left"])
        iris.draw_iris(border=True)


class TestIrisInitialization:
    def test_iris_initializes_with_pos(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        iris = Iris(frame, face_landmarks, eye_landmarks["left"])
        assert iris.pos is not None
        assert len(iris.pos) == 5

    def test_iris_initializes_with_frame(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        iris = Iris(frame, face_landmarks, eye_landmarks["left"])
        assert iris.frame is not None

    def test_iris_initializes_with_face_landmarks(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        iris = Iris(frame, face_landmarks, eye_landmarks["left"])
        assert iris.face_landmarks is not None

    def test_iris_initializes_with_id(self, mock_eye_open_frame, eye_landmarks):
        frame, face_landmarks = mock_eye_open_frame
        eye_id = eye_landmarks["left"]
        iris = Iris(frame, face_landmarks, eye_id)
        assert iris.id == eye_id
