import mmap

import torch
from torch.utils.data import Dataset


class MmapFileDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.offsets = []
        self.length = 0

        print(f"Indexing file: {file_path}...")
        with open(file_path, "rb") as f:
            self.offsets.append(f.tell())
            while f.readline():
                self.offsets.append(f.tell())

        self.offsets.pop()
        self.length = len(self.offsets)
        print(f"Indexing complete. Found {self.length} samples.")

        self.f = open(file_path, "rb")
        self.mmap_obj = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def get_batch_indices(self, indices):
        return self._get_batch(slice(indices[0], indices[-1] + 1))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._get_batch(idx)
        else:
            return self._get_single(idx)

    def _get_single(self, idx):
        start_pos = self.offsets[idx]
        end_pos = (
            self.offsets[idx + 1] if idx + 1 < self.length else self.mmap_obj.size()
        )

        line_bytes = self.mmap_obj[start_pos:end_pos]

        line = line_bytes.decode("utf-8").strip()

        parts = line.split(",")
        if len(parts) != 3:
            raise ValueError(f"Invalid line format: expected 3 columns, got {len(parts)}: {line}")
        features_str = parts[0]
        original_result = float(parts[1])
        win_prob_str = parts[2]

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

        win_prob = float(win_prob_str)

        win_prob_tensor = torch.tensor(win_prob, dtype=torch.float32)
        original_result_tensor = torch.tensor(original_result, dtype=torch.float32)
        return (
            features,
            (win_prob_tensor, original_result_tensor),
        )

    def _get_batch(self, idx_slice):
        start_idx = idx_slice.start if idx_slice.start is not None else 0
        stop_idx = idx_slice.stop if idx_slice.stop is not None else self.length
        step = idx_slice.step if idx_slice.step is not None else 1

        indices = list(range(start_idx, stop_idx, step))
        batch_size = len(indices)

        batch_features = torch.zeros((batch_size, 768), dtype=torch.float32)
        batch_win_probs = torch.zeros(batch_size, dtype=torch.float32)
        batch_original_results = torch.zeros(batch_size, dtype=torch.float32)

        for i, idx in enumerate(indices):
            start_pos = self.offsets[idx]
            end_pos = (
                self.offsets[idx + 1] if idx + 1 < self.length else self.mmap_obj.size()
            )

            line_bytes = self.mmap_obj[start_pos:end_pos]
            line = line_bytes.decode("utf-8").strip()

            parts = line.split(",")
            if len(parts) != 3:
                raise ValueError(f"Invalid line format: expected 3 columns, got {len(parts)}: {line}")
            features_part = parts[0]
            original_result = float(parts[1])
            win_prob_part = parts[2]

            feature_idx = 0
            current_num = ""

            for ch in features_part:
                if ch.isdigit():
                    current_num += ch
                elif ch == "X":
                    zero_count = int(current_num) if current_num else 0
                    feature_idx += zero_count
                    batch_features[i, feature_idx] = 1.0
                    feature_idx += 1
                    current_num = ""
                else:
                    raise ValueError(f"Unexpected character '{ch}' in features string")

            if current_num:
                zero_count = int(current_num)
                feature_idx += zero_count

            assert feature_idx == 768, f"feature_idx {feature_idx} != 768"
            win_prob = float(win_prob_part)
            batch_win_probs[i] = win_prob
            batch_original_results[i] = original_result

        return batch_features, (batch_win_probs, batch_original_results)

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "mmap_obj"):
            self.mmap_obj.close()
            delattr(self, "mmap_obj")
        if hasattr(self, "f"):
            self.f.close()
            delattr(self, "f")
