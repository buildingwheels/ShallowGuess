import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PtDataset(Dataset):
    def __init__(self, file_path, dtype=torch.float32):
        self.file_path = file_path
        self.dtype = dtype
        self.features = None
        self.labels = None

        if os.path.isdir(file_path):
            chunk_files = sorted(
                [f for f in os.listdir(file_path) if f.endswith(".pt")]
            )
            if not chunk_files:
                raise ValueError(f"No .pt files found in {file_path}")

            all_features = []
            all_labels = []

            for chunk_file in chunk_files:
                chunk_path = os.path.join(file_path, chunk_file)
                data = torch.load(chunk_path, weights_only=True)
                all_features.append(data["features"])
                all_labels.append(data["labels"])

            self.features = torch.cat(all_features, dim=0).to(dtype)
            self.labels = torch.cat(all_labels, dim=0).to(dtype)
        else:
            data = torch.load(file_path, weights_only=True)
            self.features = data["features"].to(dtype)
            self.labels = data["labels"].to(dtype)

        self.length = len(self.features)
        print(f"Loaded {self.length} samples from {file_path}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def close(self):
        if self.features is not None:
            del self.features
            self.features = None
        if self.labels is not None:
            del self.labels
            self.labels = None
