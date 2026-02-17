import mmap

import numpy as np
import torch
from torch.utils.data import Dataset


def print_indexing_progress(current, total, prefix="Indexing", length=30):
    percent = 100 * (current / float(total)) if total > 0 else 0
    filled_length = int(length * current // total) if total > 0 else 0
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:.1f}%", end="", flush=True)


class MmapFileDataset(Dataset):
    def __init__(self, file_path, show_progress=True):
        self.file_path = file_path
        self.offsets = []
        self.length = 0

        with open(file_path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(0)
            self.offsets.append(f.tell())
            update_interval = 10000
            lines_since_update = 0
            while line_bytes := f.readline():
                line = line_bytes.decode("utf-8").strip()
                if line:
                    self.offsets.append(f.tell())
                lines_since_update += 1
                if show_progress and lines_since_update >= update_interval:
                    lines_since_update = 0
                    print_indexing_progress(f.tell(), file_size, prefix="Indexing")

        self.offsets.pop()
        self.length = len(self.offsets)
        if show_progress:
            print_indexing_progress(file_size, file_size, prefix="Indexing")
            print(f" Found {self.length} samples.")

        self.f = open(file_path, "rb")
        self.mmap_obj = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_pos = self.offsets[idx]
        line_end = (
            self.offsets[idx + 1] if idx + 1 < self.length else self.mmap_obj.size()
        )

        line_bytes = self.mmap_obj[start_pos:line_end]
        line = line_bytes.decode("utf-8").strip()
        parts = line.split(",")

        if len(parts) != 2:
            raise ValueError(
                f"Invalid line format: expected 2 columns, got {len(parts)}: {line}"
            )

        features_str = parts[0]
        weighted_game_result = float(parts[1])

        tokens = features_str.split("X")
        num_segments = len(tokens)
        features = np.zeros(768, dtype=np.float32)
        pos = 0

        for i, token in enumerate(tokens):
            if not token:
                zero_count = 0
            else:
                zero_count = int(token)

            if i < num_segments - 1:
                end_pos = pos + zero_count + 1
                if end_pos > 768:
                    end_pos = 768
                features[pos:end_pos] = 0.0
                if pos + zero_count < 768:
                    features[pos + zero_count] = 1.0
                pos = end_pos
            else:
                end_pos = pos + zero_count
                if end_pos > 768:
                    end_pos = 768
                features[pos:end_pos] = 0.0
                pos = end_pos

        features_tensor = torch.from_numpy(features)
        game_result_tensor = torch.tensor(weighted_game_result, dtype=torch.float32)
        return features_tensor, game_result_tensor

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "mmap_obj"):
            self.mmap_obj.close()
            delattr(self, "mmap_obj")
        if hasattr(self, "f"):
            self.f.close()
            delattr(self, "f")
