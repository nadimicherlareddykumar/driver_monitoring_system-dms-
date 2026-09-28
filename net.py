import cv2
import h5py
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models


class MobileNet(nn.Module):
    """
    Customized MobileNet architecture using PyTorch.
    """

    def __init__(self, input_shape=(224, 224, 3), num_classes=2, dropout=0.25, lr=1e-3,
                 augmentation=False, train_base=False, add_layer=False):
        super().__init__()
        try:
            candidate_weights = models.MobileNet_V2_Weights.DEFAULT
            checkpoint = Path(torch.hub.get_dir()) / 'checkpoints' / Path(candidate_weights.url).name
            # Do not turn a local inference request into an implicit network download.
            weights = candidate_weights if checkpoint.exists() else None
        except (AttributeError, TypeError):
            weights = None
        self.model = models.mobilenet_v2(weights=weights)
        self.model.classifier[1] = nn.Linear(1280, num_classes)
        self.eval()

    def load_weights(self, checkpoint):
        with h5py.File(checkpoint, 'r') as f:
            w = f['dense']['dense']['kernel:0'][:]
            b = f['dense']['dense']['bias:0'][:]
            self.model.classifier[1].weight.data = torch.from_numpy(w.T).float()
            self.model.classifier[1].bias.data = torch.from_numpy(b).float()

    def save_weights(self, checkpoint):
        """Save the classifier head in the HDF5 format consumed by ``load_weights``."""
        linear = self.model.classifier[1]
        with h5py.File(checkpoint, 'w') as f:
            dense = f.create_group('dense').create_group('dense')
            dense.create_dataset('kernel:0', data=linear.weight.detach().cpu().numpy().T)
            dense.create_dataset('bias:0', data=linear.bias.detach().cpu().numpy())

    def predict(self, rgb_image):
        """
        Predict class probabilities for RGB image tensor (1, 224, 224, 3) or numpy array.
        Matches TensorFlow model.predict output shape (1, num_classes).
        """
        if isinstance(rgb_image, torch.Tensor):
            img_np = rgb_image.detach().cpu().numpy()
        else:
            img_np = np.asarray(rgb_image)

        if img_np.ndim == 4:
            img_np = img_np[0]

        if img_np.shape[:2] != (224, 224):
            img_np = cv2.resize(img_np, (224, 224))

        img_norm = (img_np.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).numpy()

        return probs
