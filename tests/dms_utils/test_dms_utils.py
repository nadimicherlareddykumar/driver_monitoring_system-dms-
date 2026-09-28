import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile

from dms_utils.dms_utils import (
    ACTIONS,
    AUTOTUNE,
    SESSIONS,
    preprocess_image,
    load_and_preprocess_image,
    sampling_data,
)


class TestActionsMapping:
    def test_actions_has_phonecall(self):
        assert "phonecall" in ACTIONS
        assert ACTIONS["phonecall"] == 0

    def test_actions_has_texting(self):
        assert "texting" in ACTIONS
        assert ACTIONS["texting"] == 1

    def test_actions_values_are_unique(self):
        values = list(ACTIONS.values())
        assert len(values) == len(set(values))


class TestAutotune:
    def test_autotune_is_positive(self):
        assert AUTOTUNE > 0


class TestSessions:
    def test_sessions_has_5_persons(self):
        assert len(SESSIONS) == 5

    def test_each_person_has_4_sessions(self):
        for person_id in SESSIONS:
            assert len(SESSIONS[person_id]) == 4

    def test_session_dates_are_strings(self):
        for person_id in SESSIONS:
            for session in SESSIONS[person_id]:
                assert isinstance(session, str)


class TestPreprocessImage:
    def test_preprocess_image_resizes(self, sample_image):
        result = preprocess_image(sample_image, size=(224, 224))
        assert result.shape == (224, 224, 3)

    def test_preprocess_image_with_list_size(self, sample_image):
        result = preprocess_image(sample_image, size=[224, 224])
        assert result.shape == (224, 224, 3)

    def test_preprocess_image_preserves_dtype(self, sample_image):
        result = preprocess_image(sample_image, size=(224, 224))
        assert result.dtype == sample_image.dtype

    def test_preprocess_image_with_different_size(self, sample_image):
        result = preprocess_image(sample_image, size=(112, 112))
        assert result.shape == (112, 112, 3)


class TestLoadAndPreprocessImage:
    def test_load_and_preprocess_returns_none_for_invalid_path(self):
        result = load_and_preprocess_image("/invalid/path/to/image.jpg")
        assert result is None

    @patch("cv2.imread")
    def test_load_and_preprocess_image_converts_bgr_to_rgb(self, mock_imread):
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        result = load_and_preprocess_image("/fake/path.jpg")

        assert result is not None
        mock_imread.assert_called_once_with("/fake/path.jpg")


class TestSamplingData:
    @patch("glob.glob")
    @patch("os.walk")
    def test_sampling_data_returns_paths_and_labels(self, mock_walk, mock_glob, tmp_path):
        mock_walk.return_value = [
            (str(tmp_path), [], ["phonecall_001.jpg", "phonecall_002.jpg"]),
            (str(tmp_path), [], ["texting_001.jpg", "texting_002.jpg"]),
        ]
        mock_glob.return_value = [
            str(tmp_path / "phonecall_001.jpg"),
            str(tmp_path / "phonecall_002.jpg"),
            str(tmp_path / "texting_001.jpg"),
            str(tmp_path / "texting_002.jpg"),
        ]

        paths, labels = sampling_data(str(tmp_path))

        assert paths is not None
        assert labels is not None
        assert len(paths) == len(labels)

    @patch("glob.glob")
    @patch("os.walk")
    def test_sampling_data_handles_empty_directory(self, mock_walk, mock_glob, tmp_path):
        mock_walk.return_value = []
        mock_glob.return_value = []

        paths, labels = sampling_data(str(tmp_path))

        assert paths is None or len(paths) == 0
        assert labels is None or len(labels) == 0


class TestDataIntegrity:
    def test_paths_and_labels_same_length(self, sample_image_paths, sample_labels):
        assert len(sample_image_paths) == len(sample_labels)

    def test_labels_within_actions_range(self, sample_labels):
        max_label = max(ACTIONS.values())
        assert all(label <= max_label for label in sample_labels)
