import argparse
import mmap
import os

import torch
import torch.nn.functional as F
from coach_model import CoachModel
from torch.utils.data import DataLoader, Dataset

MILLION_BASE = 1000000.0


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
        print(f"Indexing complete. Found {(self.length / MILLION_BASE):.2f}M samples.")

        self.f = open(file_path, "rb")
        self.mmap_obj = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

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

        last_comma = line.rfind(",")
        features_str = line[:last_comma]
        result_str = line[last_comma + 1 :]

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

        # Handle trailing zeros
        if current_num:
            zero_count = int(current_num)
            decompressed_features.extend([0.0] * zero_count)

        assert len(decompressed_features) == 768, (
            f"Decompressed features length {len(decompressed_features)} != 768"
        )
        features = torch.tensor(decompressed_features, dtype=torch.float32)

        original_result = float(result_str)

        return features, original_result, line

    def _get_batch(self, idx_slice):
        start_idx = idx_slice.start if idx_slice.start is not None else 0
        stop_idx = idx_slice.stop if idx_slice.stop is not None else self.length
        step = idx_slice.step if idx_slice.step is not None else 1

        indices = list(range(start_idx, stop_idx, step))
        batch_size = len(indices)

        batch_features = torch.zeros((batch_size, 768), dtype=torch.float32)
        batch_original_results = torch.zeros(batch_size, dtype=torch.float32)
        batch_original_lines = []

        for i, idx in enumerate(indices):
            start_pos = self.offsets[idx]
            end_pos = (
                self.offsets[idx + 1] if idx + 1 < self.length else self.mmap_obj.size()
            )

            line_bytes = self.mmap_obj[start_pos:end_pos]
            line = line_bytes.decode("utf-8").strip()
            batch_original_lines.append(line)

            last_comma = line.rfind(",")
            features_part = line[:last_comma]
            result_part = line[last_comma + 1 :]

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
            batch_original_results[i] = float(result_part)

        return batch_features, batch_original_results, batch_original_lines

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "mmap_obj"):
            self.mmap_obj.close()
            delattr(self, "mmap_obj")
        if hasattr(self, "f"):
            self.f.close()
            delattr(self, "f")


def mmap_collate_fn(batch):
    if not batch:
        return torch.tensor([]), torch.tensor([]), []

    features, original_results, lines = zip(*batch)
    batch_features = torch.stack(features)
    batch_original_results = torch.tensor(original_results, dtype=torch.float32)

    return batch_features, batch_original_results, list(lines)


def extract_win_probability(logits):
    probabilities = F.softmax(logits, dim=1)
    win_prob = probabilities[:, 2] + 0.5 * probabilities[:, 1]
    return win_prob


def process_file(input_file, output_file, model, device, batch_size=1024):
    dataset = MmapFileDataset(input_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        persistent_workers=False,
        collate_fn=mmap_collate_fn,
    )

    model.eval()
    total_samples = 0

    with open(output_file, "w") as out_f:
        with torch.no_grad():
            for batch_idx, (
                batch_features,
                batch_original_results,
                batch_lines,
            ) in enumerate(dataloader):
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                win_probs = extract_win_probability(outputs)

                for i, line in enumerate(batch_lines):
                    win_prob = win_probs[i].item()
                    original_result = batch_original_results[i].item()

                    last_comma = line.rfind(",")
                    features_part = line[:last_comma]
                    new_line = f"{features_part},{original_result},{win_prob:.6f}"
                    out_f.write(new_line + "\n")

                total_samples += len(batch_features)
                if batch_idx % 100 == 0:
                    print(f"Processed {total_samples} samples...")

    dataset.close()
    print(f"Completed processing {input_file}. Total samples: {total_samples}")
    return total_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to trained coach model (.pth file)")
    parser.add_argument(
        "hidden_layer_size", type=int, help="Hidden layer size of the model"
    )
    parser.add_argument("input_dir", help="Directory containing input CSV files")
    parser.add_argument("output_dir", help="Directory to save annotated CSV files")
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = CoachModel(args.hidden_layer_size).to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True),
        strict=True,
    )
    print(f"Loaded model from {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)]
    print(f"Found {len(input_files)} input files")

    total_processed = 0
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(args.output_dir, filename)

        print(f"\nProcessing {filename}...")
        processed = process_file(
            input_file, output_file, model, device, args.batch_size
        )
        total_processed += processed

    print(f"\nAnnotation complete! Total samples processed: {total_processed}")
    print(f"Output saved to: {args.output_dir}")
