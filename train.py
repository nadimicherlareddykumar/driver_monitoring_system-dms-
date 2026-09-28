"""Train the two-class driver-action classifier with PyTorch."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dms_utils.dms_utils import SESSIONS, sampling_data
from net import MobileNet


class DriverActionDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]))
        if image is None:
            raise FileNotFoundError(f"Unable to read training image: {self.paths[index]}")
        image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
        image = (image.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor, int(self.labels[index])


def session_split(paths, labels):
    """Hold out person 5, matching the original split protocol."""
    held_out_sessions = SESSIONS[5]
    test_mask = np.array([any(session in path for session in held_out_sessions) for path in paths])
    if not test_mask.any() or test_mask.all():
        raise ValueError("The session split needs both held-out and training samples")
    return paths[~test_mask], paths[test_mask], labels[~test_mask], labels[test_mask]


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model.model(images.to(device))
            correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
            total += labels.numel()
    return correct / total if total else 0.0


def train(args):
    paths, labels = sampling_data(args.data_path)
    if len(paths) == 0:
        raise ValueError("No labelled images found in data_path")

    if args.trainer == 'split':
        train_paths, test_paths, train_labels, test_labels = session_split(paths, labels)
    else:
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            paths, labels, test_size=0.2, random_state=42, stratify=labels
        )

    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    train_loader = DataLoader(DriverActionDataset(train_paths, train_labels), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(DriverActionDataset(valid_paths, valid_labels), batch_size=args.batch_size)
    test_loader = DataLoader(DriverActionDataset(test_paths, test_labels), batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        for images, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model.model(images.to(device))
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * batch_labels.size(0)
        validation_accuracy = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={loss_total / len(train_loader.dataset):.4f}, val_accuracy={validation_accuracy:.3f}")

    output_path = Path(args.save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(output_path)
    print(f"Saved classifier weights to {output_path}")
    print(f"Test accuracy: {evaluate(model, test_loader, device):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='DMD dataset root directory')
    parser.add_argument('--save-path', default='models/model_split.h5', help='HDF5 output for inference')
    parser.add_argument('--trainer', choices=('random', 'split'), default='random', help='Dataset split strategy')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
