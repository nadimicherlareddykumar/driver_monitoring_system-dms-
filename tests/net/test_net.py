import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import h5py
import tempfile
import os

from net import MobileNet


class TestMobileNetInit:
    def test_mobilenet_init_default_params(self):
        model = MobileNet()
        assert model.model is not None
        assert model.model.classifier[1].out_features == 2

    def test_mobilenet_init_custom_num_classes(self):
        model = MobileNet(num_classes=4)
        assert model.model.classifier[1].out_features == 4

    def test_mobilenet_init_custom_dropout(self):
        model = MobileNet(dropout=0.5)
        assert model is not None

    def test_mobilenet_init_custom_input_shape(self):
        model = MobileNet(input_shape=(160, 160, 3))
        assert model is not None


class TestMobileNetPredict:
    def test_predict_single_image_numpy(self):
        model = MobileNet()
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        probs = model.predict(image)
        assert probs.shape == (1, 2)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)

    def test_predict_single_image_with_batch_dim(self):
        model = MobileNet()
        image = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
        probs = model.predict(image)
        assert probs.shape == (1, 2)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)

    def test_predict_resizes_non_standard_size(self):
        model = MobileNet()
        image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        probs = model.predict(image)
        assert probs.shape == (1, 2)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)

    def test_predict_returns_tensor_on_tensor_input(self):
        import torch
        model = MobileNet()
        image = torch.randint(0, 255, (224, 224, 3), dtype=torch.float32)
        probs = model.predict(image)
        assert probs.shape == (1, 2)

    def test_prediction_sum_to_one(self):
        model = MobileNet()
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        probs = model.predict(image)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_prediction_all_positive(self):
        model = MobileNet()
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        probs = model.predict(image)
        assert all(p > 0 for p in probs[0])


class TestMobileNetLoadWeights:
    def test_load_weights_with_valid_file(self, tmp_path):
        model = MobileNet()

        with h5py.File(tmp_path / "weights.h5", "w") as f:
            dense = f.create_group("dense/dense")
            dense.create_dataset("kernel:0", data=np.random.randn(1280, 2))
            dense.create_dataset("bias:0", data=np.random.randn(2))

        model.load_weights(str(tmp_path / "weights.h5"))

    def test_load_weights_file_not_found(self):
        model = MobileNet()
        with pytest.raises(OSError):
            model.load_weights("/nonexistent/path/weights.h5")


class TestMobileNetInference:
    def test_model_is_eval_mode(self):
        model = MobileNet()
        assert not model.model.training

    def test_model_predict_does_not_modify_input(self):
        model = MobileNet()
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        original = image.copy()
        _ = model.predict(image)
        assert np.array_equal(image, original)
