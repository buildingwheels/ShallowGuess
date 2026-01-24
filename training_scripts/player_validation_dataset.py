import mmap

import torch
from torch.utils.data import Dataset


class PlayerValidationDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.offsets = []
        self.length = 0

        print(f"Indexing validation file: {file_path}...")
        with open(file_path, "rb") as f:
            self.offsets.append(f.tell())
            while f.readline():
                self.offsets.append(f.tell())

        self.offsets.pop()
        self.length = len(self.offsets)
        print(f"Validation indexing complete. Found {self.length} samples.")

        self.f = open(file_path, "rb")
        self.mmap_obj = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self._get_single(idx)

    def _get_single(self, idx):
        start_pos = self.offsets[idx]
        end_pos = (
            self.offsets[idx + 1] if idx + 1 < self.length else self.mmap_obj.size()
        )

        line_bytes = self.mmap_obj[start_pos:end_pos]

        line = line_bytes.decode("utf-8").strip()

        parts = line.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid validation line format: expected 2 columns (features,result), got {len(parts)}: {line}")
        features_str = parts[0]
        result_str = parts[1]

        decompressed_features = []
        current_num = ""

        for ch in features_str:
            if ch.isdigit():
                current_num += ch
            elif ch == "X":
                zero_count = int(current_num) if current_num else 0
                decompressed_features.extend([0.0] * zero_count)
                decompressed_features.append(1.0)
                current_num = ""
            else:
                raise ValueError(f"Unexpected character '{ch}' in features string")

        if current_num:
            zero_count = int(current_num)
            decompressed_features.extend([0.0] * zero_count)

        assert len(decompressed_features) == 768, (
            f"Decompressed features length {len(decompressed_features)} != 768"
        )
        features = torch.tensor(decompressed_features, dtype=torch.float32)

        original_result = float(result_str)
        if original_result not in [0.0, 0.5, 1.0]:
            raise ValueError(f"Invalid result {original_result}")

        target = torch.tensor(original_result, dtype=torch.float32)
        return features, target

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "mmap_obj"):
            self.mmap_obj.close()
            delattr(self, "mmap_obj")
        if hasattr(self, "f"):
            self.f.close()
            delattr(self, "f")