import argparse
import mmap
import os
import random
import signal
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ShallowGuessNetwork
from torch.utils.data import DataLoader, Dataset

HUNDRED_BASE = 100.0
MILLION_BASE = 1000000.0
SECONDS_PER_HOUR = 3600


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
        print(
            f"Indexing complete. Found {(self.length / MILLION_BASE):.2f}M samples."
        )

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
        values = line.split(",")

        decompressed_features = []
        for value in values[:-2]:
            if value.isdigit():
                decompressed_features.extend([0.0] * int(value))
            else:
                decompressed_features.append(1.0)

        assert len(decompressed_features) == 768, f"Decompressed features length {len(decompressed_features)} != 768"
        features = torch.tensor(decompressed_features, dtype=torch.float32)

        original_result = float(values[-2])

        if original_result == 0.0:
            result = torch.tensor(0, dtype=torch.long)
        elif original_result == 0.5:
            result = torch.tensor(1, dtype=torch.long)
        elif original_result == 1.0:
            result = torch.tensor(2, dtype=torch.long)
        else:
            raise ValueError(f"Invalid result {original_result}")

        position_count = int(values[-1])

        return (
            features,
            result,
            torch.tensor(position_count, dtype=torch.float32),
        )

    def _get_batch(self, idx_slice):
        start_idx = idx_slice.start if idx_slice.start is not None else 0
        stop_idx = idx_slice.stop if idx_slice.stop is not None else self.length
        step = idx_slice.step if idx_slice.step is not None else 1

        indices = list(range(start_idx, stop_idx, step))
        batch_size = len(indices)

        batch_features = torch.zeros((batch_size, 768), dtype=torch.float32)
        batch_results = torch.zeros(batch_size, dtype=torch.long)
        batch_position_counts = torch.zeros(batch_size, dtype=torch.float32)

        for i, idx in enumerate(indices):
            start_pos = self.offsets[idx]
            end_pos = (
                self.offsets[idx + 1] if idx + 1 < self.length else self.mmap_obj.size()
            )

            line_bytes = self.mmap_obj[start_pos:end_pos]
            line = line_bytes.decode("utf-8").strip()

            last_comma = line.rfind(",")
            second_last_comma = line.rfind(",", 0, last_comma)

            features_part = line[:second_last_comma]
            result_part = line[second_last_comma + 1 : last_comma]
            position_part = line[last_comma + 1 :]

            feature_idx = 0
            value_start = 0
            while value_start < len(features_part):
                next_comma = features_part.find(",", value_start)
                if next_comma == -1:
                    next_comma = len(features_part)

                value = features_part[value_start:next_comma]
                if value.isdigit():
                    feature_idx += int(value)
                else:
                    batch_features[i, feature_idx] = 1.0
                    feature_idx += 1

                value_start = next_comma + 1

            assert feature_idx == 768, f"feature_idx {feature_idx} != 768"
            original_result = float(result_part)
            if original_result == 0.0:
                batch_results[i] = 0
            elif original_result == 0.5:
                batch_results[i] = 1
            elif original_result == 1.0:
                batch_results[i] = 2
            else:
                raise ValueError(f"Invalid result {original_result}")

            position_count = int(position_part)
            batch_position_counts[i] = float(position_count)

        return batch_features, batch_results, batch_position_counts

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
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    features, results, position_counts = zip(*batch)

    batch_features = torch.stack(features)
    batch_results = torch.stack(results)
    batch_position_counts = torch.stack(position_counts)

    return batch_features, batch_results, batch_position_counts


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, weights=None):
        if weights is None:
            return F.cross_entropy(outputs, targets)

        loss_per_sample = F.cross_entropy(outputs, targets, reduction="none")
        weighted_loss_per_sample = loss_per_sample * weights

        return weighted_loss_per_sample.sum()


training_interrupted = False


def bold(text):
    """Return text with bold formatting."""
    return f"\033[1m{text}\033[0m"


def signal_handler(sig, frame):
    global training_interrupted
    if not training_interrupted:
        print("\nCtrl+C detected! Stopping training and saving model...")
        training_interrupted = True
    else:
        print("\nForce exit...")
        sys.exit(1)


def check_gradient_norm(model):
    total_norm = 0.0
    layer_norms = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm**2
            layer_norms.append((name, param_norm))

    total_norm = total_norm**0.5 if total_norm > 0 else 0.0
    return total_norm, layer_norms


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("first_hidden_layer_size", type=int)
    parser.add_argument("data_dir")
    parser.add_argument("model_export_path")
    parser.add_argument("max_epochs", type=int)
    parser.add_argument("sample_size", type=int)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--print_cycle", type=int, default=100)
    parser.add_argument("--export_cycle", type=int, default=1000)
    parser.add_argument("--existing_pth_file", default=None)
    parser.add_argument("--enable_diagnostics", action="store_true", default=False)
    parser.add_argument(
        "--max_pos_count",
        type=int,
        default=400,
        help="Max move count for weight scaling",
    )
    parser.add_argument(
        "--training_log_file",
        type=str,
        default=None,
        help="Path to the training log file",
    )
    args = parser.parse_args()

    if args.training_log_file is None:
        args.training_log_file = (
            f"temp/training_log_{args.first_hidden_layer_size}.log"
        )

    os.makedirs(os.path.dirname(args.training_log_file), exist_ok=True)

    if not os.path.exists(args.training_log_file):
        with open(args.training_log_file, "w") as f:
            f.write("Training Config:\n")
            f.write(f"Model Hidden Layers={args.first_hidden_layer_size}\n")
            f.write(f"Batch Size={args.batch_size}\n")
            f.write(f"Learning Rate={args.learning_rate}\n")
            f.write(f"Max Epochs={args.max_epochs}\n")
            f.write(f"Sample Size={args.sample_size}\n")
            f.write(f"Max Position Count={args.max_pos_count}\n\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Hidden layers: {args.first_hidden_layer_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Log file: {args.training_log_file}")

    model = ShallowGuessNetwork(args.first_hidden_layer_size).to(device)

    if args.existing_pth_file:
        model.load_state_dict(
            torch.load(args.existing_pth_file, map_location=device, weights_only=True),
            strict=False,
        )
        print(f"Loaded existing model file {args.existing_pth_file}")

    weighted_criterion = WeightedCrossEntropyLoss()

    standard_criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=1
    )

    file_list = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)]

    model.train()

    start_time = datetime.now()

    trained_files_count = {file: 0 for file in file_list}
    total_samples_trained = 0

    for epoch in range(args.max_epochs):
        if training_interrupted:
            break
        epoch_start_time = datetime.now()
        selected_files = random.sample(file_list, args.sample_size)
        num_selected_files = len(selected_files)

        for file_name in selected_files:
            trained_files_count[file_name] += 1

        epoch_loss = 0.0
        epoch_weighted_loss = 0.0
        epoch_weights_sum = 0.0
        epoch_samples_trained = 0
        batch_count = 0

        for file_idx, file_path in enumerate(selected_files):
            if training_interrupted:
                break

            dataset = MmapFileDataset(file_path)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                persistent_workers=False,
                collate_fn=mmap_collate_fn,
            )

            file_start_time = datetime.now()
            try:
                for batch_idx, (
                    batch_features,
                    batch_results,
                    batch_position_counts,
                ) in enumerate(dataloader):
                    if training_interrupted:
                        break

                    batch_start_time = datetime.now()
                    batch_features = batch_features.to(device)
                    batch_results = batch_results.to(device)
                    batch_position_counts = batch_position_counts.to(device)

                    batch_weights = torch.tensor(
                        [
                            pos.item() / args.max_pos_count
                            for pos in batch_position_counts
                        ],
                        device=device,
                    )

                    optimizer.zero_grad()
                    outputs = model(batch_features)

                    weighted_loss = weighted_criterion(
                        outputs, batch_results, batch_weights
                    )
                    weighted_loss.backward()

                    with torch.no_grad():
                        standard_loss = standard_criterion(outputs, batch_results)

                    optimizer.step()

                    batch_loss = standard_loss.item()
                    batch_weighted_loss = weighted_loss.item()

                    epoch_loss += batch_loss
                    epoch_weighted_loss += batch_weighted_loss
                    epoch_weights_sum += batch_weights.sum().item()

                    samples_count = len(batch_features)
                    epoch_samples_trained += samples_count
                    total_samples_trained += samples_count

                    batch_time = (datetime.now() - batch_start_time).total_seconds()
                    samples_per_second = (
                        samples_count / batch_time if batch_time > 0 else 0
                    )

                    progress = (
                        epoch_samples_trained
                        / (len(dataset) * num_selected_files)
                        * HUNDRED_BASE
                    )
                    batch_count += 1

                    if batch_idx > 1 and batch_idx % args.print_cycle == 0:
                        grad_norm, _ = check_gradient_norm(model)

                        print(
                            f"{bold('Epoch')} {epoch}/{args.max_epochs} "
                            f"{bold('File')} {file_idx + 1}/{num_selected_files} "
                            f"{bold('Batch')} {batch_idx + 1}/{len(dataloader)} "
                            f"{bold('Samples')}: {(total_samples_trained / MILLION_BASE):.2f}M "
                            f"{bold('Loss')}: {batch_loss:.4f} "
                            f"{bold('Avg Weighted Loss')}: {(epoch_weighted_loss / max(batch_count, 1)):.4f} "
                            f"{bold('Grad Norm')}: {grad_norm:.4f} "
                            f"{bold('Batch Time')}: {batch_time:.3f}s "
                            f"{bold('Samples/sec')}: {samples_per_second:.0f} "
                            f"{bold('LR')}: {optimizer.param_groups[0]['lr']:.6f}"
                        )

                    if batch_idx > 1 and batch_idx % args.export_cycle == 0:
                        model.eval()
                        torch.save(
                            model.state_dict(),
                            f"{args.model_export_path}/{model.pub_name()}.pth",
                        )

                        current_avg_weighted_loss = epoch_weighted_loss / max(
                            batch_count, 1
                        )

                        with open(args.training_log_file, "a") as f:
                            epoch_time = (
                                datetime.now() - epoch_start_time
                            ).total_seconds()
                            total_time = (datetime.now() - start_time).total_seconds()
                            f.write(
                                f"Timestamp={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            )
                            f.write(f"Epoch={epoch}/{args.max_epochs}\n")
                            f.write(f"Batch={batch_idx + 1}/{len(dataloader)}\n")
                            f.write(
                                f"Samples (M)={total_samples_trained / MILLION_BASE:.2f}\n"
                            )
                            f.write(f"Loss={batch_loss:.6f}\n")
                            f.write(
                                f"Avg Weighted Loss={current_avg_weighted_loss:.6f}\n"
                            )
                            f.write(f"Batch Weighted Loss={batch_weighted_loss:.6f}\n")
                            f.write(
                                f"Epoch Time (h)={epoch_time / SECONDS_PER_HOUR:.3f}\n"
                            )
                            f.write(
                                f"Total Time (h)={total_time / SECONDS_PER_HOUR:.3f}\n"
                            )
                            f.write(
                                f"Learning Rate={optimizer.param_groups[0]['lr']:.6f}\n"
                            )
                            f.write(f"Batch Time (s)={batch_time:.3f}\n")
                            f.write(f"Samples/sec={samples_per_second:.0f}\n\n")

                        print(
                            f"[SAVED] {args.model_export_path}/{model.pub_name()}.pth"
                        )
                        model.train()
            finally:
                file_time = (datetime.now() - file_start_time).total_seconds()
                print(
                    f"[FILE] {file_idx + 1}/{num_selected_files} completed in {file_time:.1f}s"
                )

                dataset.close()
                del dataset
                del dataloader

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        avg_weighted_loss = epoch_weighted_loss / max(batch_count, 1)
        avg_weight = epoch_weights_sum / max(epoch_samples_trained, 1)
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        total_time = (datetime.now() - start_time).total_seconds()

        scheduler.step(avg_weighted_loss)

        samples_per_sec_epoch = (
            epoch_samples_trained / epoch_time if epoch_time > 0 else 0
        )

        with open(args.training_log_file, "a") as f:
            f.write("--- Epoch Summary ---\n")
            f.write(f"Epoch={epoch}/{args.max_epochs}\n")
            f.write(f"Avg Loss={avg_epoch_loss:.6f}\n")
            f.write(f"Avg Weighted Loss={avg_weighted_loss:.6f}\n")
            f.write(f"Avg Weight={avg_weight:.6f}\n")
            f.write(f"Epoch Time (h)={epoch_time / SECONDS_PER_HOUR:.3f}\n")
            f.write(f"Total Time (h)={total_time / SECONDS_PER_HOUR:.3f}\n")
            f.write(f"Avg Samples/sec={samples_per_sec_epoch:.0f}\n")
            f.write(f"Learning Rate={optimizer.param_groups[0]['lr']:.6f}\n\n")

        print(
            f"--- Epoch {epoch}/{args.max_epochs} Completed --- \n"
            f"{bold('Loss')}: {avg_epoch_loss:.4f}\n"
            f"{bold('Avg Weighted Loss')}: {avg_weighted_loss:.4f}\n"
            f"{bold('Avg Weight')}: {avg_weight:.4f}\n"
            f"{bold('Epoch Time')}: {epoch_time / SECONDS_PER_HOUR:.2f}h\n"
            f"{bold('Total Time')}: {total_time / SECONDS_PER_HOUR:.2f}h\n"
            f"{bold('Avg Samples/sec')}: {samples_per_sec_epoch:.0f}\n"
            f"{bold('LR')}: {optimizer.param_groups[0]['lr']:.6f}\n\n"
        )

    if training_interrupted:
        print("[INTERRUPTED] Saving final model...")
    else:
        print("[COMPLETE] Saving final model...")

    model.eval()
    torch.save(model.state_dict(), f"{args.model_export_path}/{model.pub_name()}.pth")
    print(f"[FINAL] {args.model_export_path}/{model.pub_name()}.pth")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_training_time = (datetime.now() - start_time).total_seconds()
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = total_training_time % 60

    print("=== Training Summary ===\n")
    print(f"Total samples trained: {total_samples_trained:,}")
    print(f"Total training time: {hours}h {minutes}m {seconds:.1f}s")
    print(
        f"Average samples per second: {total_samples_trained / total_training_time:.1f}"
    )

    if training_interrupted:
        print("[DONE] Training stopped successfully.")
        sys.exit(0)
