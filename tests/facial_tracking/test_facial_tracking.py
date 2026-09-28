import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from facial_tracking.facialTracking import FacialTracker


class TestFacialTrackerInit:
    def test_facial_tracker_initializes(self):
        tracker = FacialTracker()
        assert tracker.fm is not None
        assert tracker.left_eye is None
        assert tracker.right_eye is None
        assert tracker.lips is None
        assert tracker.left_eye_closed_frames == 0
        assert tracker.right_eye_closed_frames == 0


class TestEyesStatus:
    @patch("facial_tracking.facialTracking.FaceMesh")
    def test_eyes_status_center(self, MockFaceMesh, mock_eye_open_frame, eye_landmarks):
        MockFaceMesh.return_value.mesh_result.multi_face_landmarks = [MagicMock()]
        mock_face_landmarks = MagicMock()
        for i, eye_id in enumerate(eye_landmarks["left"][:4]):
            mock_face_landmarks.landmark[eye_id].x = 0.35 + i * 0.005
            mock_face_landmarks.landmark[eye_id].y = 0.4 + (0.02 if i >= 2 else 0)
        for i, eye_id in enumerate(eye_landmarks["right"][:4]):
            mock_face_landmarks.landmark[eye_id].x = 0.6 + i * 0.005
            mock_face_landmarks.landmark[eye_id].y = 0.4 + (0.02 if i >= 2 else 0)
        mock_face_landmarks.landmark[eye_landmarks["left"][-5]].x = 0.35
        mock_face_landmarks.landmark[eye_landmarks["left"][-5]].y = 0.4
        mock_face_landmarks.landmark[eye_landmarks["right"][-5]].x = 0.6
        mock_face_landmarks.landmark[eye_landmarks["right"][-5]].y = 0.4

        with patch.object(FacialTracker, "process_frame"):
            tracker = FacialTracker()
            tracker.fm.mesh_result.multi_face_landmarks = [mock_face_landmarks]
            tracker._check_eyes_status()

    @patch("facial_tracking.facialTracking.FaceMesh")
    def test_eyes_status_left(self, MockFaceMesh, mock_gaze_left_frame, eye_landmarks):
        MockFaceMesh.return_value.mesh_result.multi_face_landmarks = [MagicMock()]
        tracker = FacialTracker()
        tracker.left_eye_closed_frames = 0
        tracker.right_eye_closed_frames = 0
        tracker.left_eye = MagicMock()
        tracker.right_eye = MagicMock()
        tracker.left_eye.gaze_left.return_value = True
        tracker.left_eye.gaze_right.return_value = False
        tracker.left_eye.eye_closed.return_value = False
        tracker.right_eye.gaze_left.return_value = True
        tracker.right_eye.gaze_right.return_value = False
        tracker.right_eye.eye_closed.return_value = False

        tracker._check_eyes_status()
        assert tracker.eyes_status == "gazing left"

    @patch("facial_tracking.facialTracking.FaceMesh")
    def test_eyes_status_right(self, MockFaceMesh, mock_gaze_right_frame, eye_landmarks):
        MockFaceMesh.return_value.mesh_result.multi_face_landmarks = [MagicMock()]
        tracker = FacialTracker()
        tracker.left_eye_closed_frames = 0
        tracker.right_eye_closed_frames = 0
        tracker.left_eye = MagicMock()
        tracker.right_eye = MagicMock()
        tracker.left_eye.gaze_left.return_value = False
        tracker.left_eye.gaze_right.return_value = True
        tracker.left_eye.eye_closed.return_value = False
        tracker.right_eye.gaze_left.return_value = False
        tracker.right_eye.gaze_right.return_value = True
        tracker.right_eye.eye_closed.return_value = False

        tracker._check_eyes_status()
        assert tracker.eyes_status == "gazing right"


class TestEyeClosedFramesCounter:
    @patch("facial_tracking.facialTracking.FaceMesh")
    def test_eye_closed_frames_increments(self, MockFaceMesh):
        tracker = FacialTracker()
        tracker.left_eye = MagicMock()
        tracker.right_eye = MagicMock()
        tracker.left_eye.eye_closed.return_value = True
        tracker.right_eye.eye_closed.return_value = True

        tracker._check_eyes_status()
        assert tracker.left_eye_closed_frames == 1
        assert tracker.right_eye_closed_frames == 1

    @patch("facial_tracking.facialTracking.FaceMesh")
    def test_eye_closed_frames_resets_when_open(self, MockFaceMesh):
        tracker = FacialTracker()
        tracker.left_eye_closed_frames = 5
        tracker.left_eye = MagicMock()
        tracker.right_eye = MagicMock()
        tracker.left_eye.eye_closed.return_value = False
        tracker.right_eye.eye_closed.return_value = False

        tracker._check_eyes_status()
        assert tracker.left_eye_closed_frames == 0


class TestYawnStatus:
    @patch("facial_tracking.facialTracking.FaceMesh")
    def test_yawn_status_set_when_mouth_open(self, MockFaceMesh, mock_mouth_open_frame, lips_landmarks):
        MockFaceMesh.return_value.mesh_result.multi_face_landmarks = [MagicMock()]
        tracker = FacialTracker()
        tracker.lips = MagicMock()
        tracker.lips.mouth_open.return_value = True

        tracker._check_yawn_status()
        assert tracker.yawn_status == "yawning"

    @patch("facial_tracking.facialTracking.FaceMesh")
    def test_yawn_status_empty_when_mouth_closed(self, MockFaceMesh, mock_mouth_closed_frame, lips_landmarks):
        MockFaceMesh.return_value.mesh_result.multi_face_landmarks = [MagicMock()]
        tracker = FacialTracker()
        tracker.lips = MagicMock()
        tracker.lips.mouth_open.return_value = False

        tracker._check_yawn_status()
        assert tracker.yawn_status == "Normal"


class TestLeftRightEyeClosed:
    @patch("facial_tracking.facialTracking.FaceMesh")
    def test_left_eye_closed_threshold(self, MockFaceMesh):
        tracker = FacialTracker()
        tracker.left_eye_closed_frames = 15

        assert tracker._left_eye_closed() is True
        assert tracker._left_eye_closed(threshold=20) is False

    @patch("facial_tracking.facialTracking.FaceMesh")
    def test_right_eye_closed_threshold(self, MockFaceMesh):
        tracker = FacialTracker()
        tracker.right_eye_closed_frames = 15

        assert tracker._right_eye_closed() is True
        assert tracker._right_eye_closed(threshold=20) is False


class TestProcessFrame:
    @patch("facial_tracking.facialTracking.FaceMesh")
    def test_process_frame_sets_detected_true(self, MockFaceMesh, mock_face_landmarks):
        mock_mesh = MagicMock()
        mock_mesh.mesh_result.multi_face_landmarks = [mock_face_landmarks]
        MockFaceMesh.return_value = mock_mesh

        tracker = FacialTracker()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        tracker.process_frame(frame)
        assert tracker.detected is True

    @patch("facial_tracking.facialTracking.FaceMesh")
    def test_process_frame_sets_detected_false(self, MockFaceMesh):
        mock_mesh = MagicMock()
        mock_mesh.mesh_result.multi_face_landmarks = []
        MockFaceMesh.return_value = mock_mesh

        tracker = FacialTracker()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        tracker.process_frame(frame)
        assert tracker.detected is False
