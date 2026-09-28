import numpy as np
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = [200, 200, 200]
    return frame


@pytest.fixture
def mock_face_landmarks():
    landmarks = MagicMock()
    landmark_list = []

    for i in range(468):
        lm = MagicMock()
        lm.x = 0.5 + (np.random.random() - 0.5) * 0.1
        lm.y = 0.5 + (np.random.random() - 0.5) * 0.1
        lm.z = 0.0
        landmark_list.append(lm)

    landmarks.landmark = landmark_list
    return landmarks


@pytest.fixture
def eye_landmarks():
    from facial_tracking.conf import LEFT_EYE, RIGHT_EYE
    return {"left": LEFT_EYE, "right": RIGHT_EYE}


@pytest.fixture
def lips_landmarks():
    from facial_tracking.conf import LIPS
    return LIPS


@pytest.fixture
def mock_eye_open_frame(mock_frame, mock_face_landmarks, eye_landmarks):
    h, w = mock_frame.shape[:2]

    left_eye_ids = eye_landmarks["left"][:4]
    right_eye_ids = eye_landmarks["right"][:4]

    for i, lm_id in enumerate(left_eye_ids[:4]):
        mock_face_landmarks.landmark[lm_id].x = 0.35 + i * 0.005
        mock_face_landmarks.landmark[lm_id].y = 0.4 + (0 if i < 2 else 0.02)

    for i, lm_id in enumerate(right_eye_ids[:4]):
        mock_face_landmarks.landmark[lm_id].x = 0.6 + i * 0.005
        mock_face_landmarks.landmark[lm_id].y = 0.4 + (0 if i < 2 else 0.02)

    return mock_frame, mock_face_landmarks


@pytest.fixture
def mock_eye_closed_frame(mock_frame, mock_face_landmarks, eye_landmarks):
    h, w = mock_frame.shape[:2]

    left_eye_ids = eye_landmarks["left"][:4]
    right_eye_ids = eye_landmarks["right"][:4]

    for i, lm_id in enumerate(left_eye_ids[:4]):
        mock_face_landmarks.landmark[lm_id].x = 0.35 + i * 0.005
        mock_face_landmarks.landmark[lm_id].y = 0.4

    for i, lm_id in enumerate(right_eye_ids[:4]):
        mock_face_landmarks.landmark[lm_id].x = 0.6 + i * 0.005
        mock_face_landmarks.landmark[lm_id].y = 0.4

    return mock_frame, mock_face_landmarks


@pytest.fixture
def mock_gaze_left_frame(mock_frame, mock_face_landmarks, eye_landmarks):
    h, w = mock_frame.shape[:2]

    left_eye_ids = eye_landmarks["left"]
    right_eye_ids = eye_landmarks["right"]

    mock_face_landmarks.landmark[left_eye_ids[0]].x = 0.4
    mock_face_landmarks.landmark[left_eye_ids[0]].y = 0.4
    mock_face_landmarks.landmark[left_eye_ids[1]].x = 0.3
    mock_face_landmarks.landmark[left_eye_ids[1]].y = 0.4
    mock_face_landmarks.landmark[left_eye_ids[2]].x = 0.35
    mock_face_landmarks.landmark[left_eye_ids[2]].y = 0.38
    mock_face_landmarks.landmark[left_eye_ids[3]].x = 0.35
    mock_face_landmarks.landmark[left_eye_ids[3]].y = 0.42

    mock_face_landmarks.landmark[right_eye_ids[0]].x = 0.65
    mock_face_landmarks.landmark[right_eye_ids[0]].y = 0.4
    mock_face_landmarks.landmark[right_eye_ids[1]].x = 0.55
    mock_face_landmarks.landmark[right_eye_ids[1]].y = 0.4
    mock_face_landmarks.landmark[right_eye_ids[2]].x = 0.6
    mock_face_landmarks.landmark[right_eye_ids[2]].y = 0.38
    mock_face_landmarks.landmark[right_eye_ids[3]].x = 0.6
    mock_face_landmarks.landmark[right_eye_ids[3]].y = 0.42

    return mock_frame, mock_face_landmarks


@pytest.fixture
def mock_gaze_right_frame(mock_frame, mock_face_landmarks, eye_landmarks):
    h, w = mock_frame.shape[:2]

    left_eye_ids = eye_landmarks["left"]
    right_eye_ids = eye_landmarks["right"]

    mock_face_landmarks.landmark[left_eye_ids[0]].x = 0.4
    mock_face_landmarks.landmark[left_eye_ids[0]].y = 0.4
    mock_face_landmarks.landmark[left_eye_ids[1]].x = 0.3
    mock_face_landmarks.landmark[left_eye_ids[1]].y = 0.4
    mock_face_landmarks.landmark[left_eye_ids[2]].x = 0.35
    mock_face_landmarks.landmark[left_eye_ids[2]].y = 0.38
    mock_face_landmarks.landmark[left_eye_ids[3]].x = 0.35
    mock_face_landmarks.landmark[left_eye_ids[3]].y = 0.42

    mock_face_landmarks.landmark[right_eye_ids[0]].x = 0.65
    mock_face_landmarks.landmark[right_eye_ids[0]].y = 0.4
    mock_face_landmarks.landmark[right_eye_ids[1]].x = 0.55
    mock_face_landmarks.landmark[right_eye_ids[1]].y = 0.4
    mock_face_landmarks.landmark[right_eye_ids[2]].x = 0.6
    mock_face_landmarks.landmark[right_eye_ids[2]].y = 0.38
    mock_face_landmarks.landmark[right_eye_ids[3]].x = 0.6
    mock_face_landmarks.landmark[right_eye_ids[3]].y = 0.42

    return mock_frame, mock_face_landmarks


@pytest.fixture
def mock_mouth_open_frame(mock_frame, mock_face_landmarks, lips_landmarks):
    h, w = mock_frame.shape[:2]

    mock_face_landmarks.landmark[lips_landmarks[0]].x = 0.45
    mock_face_landmarks.landmark[lips_landmarks[0]].y = 0.55
    mock_face_landmarks.landmark[lips_landmarks[1]].x = 0.55
    mock_face_landmarks.landmark[lips_landmarks[1]].y = 0.55
    mock_face_landmarks.landmark[lips_landmarks[2]].x = 0.5
    mock_face_landmarks.landmark[lips_landmarks[2]].y = 0.58
    mock_face_landmarks.landmark[lips_landmarks[3]].x = 0.5
    mock_face_landmarks.landmark[lips_landmarks[3]].y = 0.65

    return mock_frame, mock_face_landmarks


@pytest.fixture
def mock_mouth_closed_frame(mock_frame, mock_face_landmarks, lips_landmarks):
    h, w = mock_frame.shape[:2]

    mock_face_landmarks.landmark[lips_landmarks[0]].x = 0.45
    mock_face_landmarks.landmark[lips_landmarks[0]].y = 0.55
    mock_face_landmarks.landmark[lips_landmarks[1]].x = 0.55
    mock_face_landmarks.landmark[lips_landmarks[1]].y = 0.55
    mock_face_landmarks.landmark[lips_landmarks[2]].x = 0.5
    mock_face_landmarks.landmark[lips_landmarks[2]].y = 0.56
    mock_face_landmarks.landmark[lips_landmarks[3]].x = 0.5
    mock_face_landmarks.landmark[lips_landmarks[3]].y = 0.57

    return mock_frame, mock_face_landmarks


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_paths():
    return np.array([
        "/path/to/phonecall_001.jpg",
        "/path/to/phonecall_002.jpg",
        "/path/to/texting_001.jpg",
        "/path/to/texting_002.jpg",
    ])


@pytest.fixture
def sample_labels():
    return np.array([0, 0, 1, 1])
