import argparse
import os
import random
import sys
from datetime import datetime
from math import cos, pi

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_optimizer import Ranger
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

HUNDRED_BASE = 100.0
MILLION_BASE = 1000000.0
SECONDS_PER_HOUR = 3600


class WarmupCosineScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        cosine_factor = 0.5 * (1 + cos(pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_factor
            for base_lr in self.base_lrs
        ]


def bold(text):
    return f"\033[1m{text}\033[0m"


def print_progress_bar(
    iteration, total, prefix="", suffix="", decimals=1, length=50, fill="█"
):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="", flush=True)
    if iteration == total:
        print()


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


def evaluate_validation(
    model,
    criterion,
    device,
    batch_size,
    validation_dataset,
):
    dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=False,
    )

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_samples = 0

    val_start_time = datetime.now()
    total_batches = len(dataloader)

    print(f"\n{bold('Running validation...')}")
    with torch.no_grad():
        for batch_idx, (batch_features, batch_targets) in enumerate(dataloader):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_targets)

            total_loss += loss.item() * len(batch_features)
            total_samples += len(batch_features)

            print_progress_bar(
                batch_idx + 1,
                total_batches,
                prefix="Validating:",
                suffix=f"Batch {batch_idx + 1}/{total_batches}",
                length=30,
            )

    val_time = (datetime.now() - val_start_time).total_seconds()
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    samples_per_sec = total_samples / val_time if val_time > 0 else 0

    print(
        f"\r{bold('Validation')}: {bold('Loss')}={avg_loss:.6f} "
        f"{bold('Samples')}={total_samples} "
        f"{bold('Samples/sec')}={samples_per_sec:.0f} "
        f"{bold('Time')}={val_time:.1f}s"
    )

    model.train(was_training)

    del dataloader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hidden_layer_size", type=int)
    parser.add_argument("data_dir")
    parser.add_argument("validation_file")
    parser.add_argument("model_export_path")
    parser.add_argument("max_epochs", type=int)
    parser.add_argument("sample_size", type=int)
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--existing_pth_file", default=None)
    parser.add_argument("--enable_diagnostics", action="store_true", default=False)

    parser.add_argument(
        "--training_log_file",
        type=str,
        default=None,
        help="Path to the training log file",
    )
    parser.add_argument(
        "--fine_tuning",
        action="store_true",
        default=False,
        help="Use fine-tuning mode (SGD with momentum) instead of standard training (AdamW)",
    )
    parser.add_argument(
        "--use_pt_format",
        action="store_true",
        default=False,
        help="Use pre-converted .pt format instead of text format (faster loading)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=16,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "total_steps",
        type=int,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--save_cycle",
        type=int,
        default=0,
        help="Save model every N batches (0 = disabled)",
    )

    args = parser.parse_args()

    if args.use_pt_format:
        from pt_dataset import PtDataset as MmapFileDataset
    else:
        from player_dataset import MmapFileDataset
    from player_model import PlayerModel

    Model = PlayerModel
    criterion = nn.MSELoss()
    log_suffix = "_player"

    if args.training_log_file is None:
        args.training_log_file = (
            f"temp/training_log{log_suffix}_{args.hidden_layer_size}.log"
        )

    os.makedirs(os.path.dirname(args.training_log_file), exist_ok=True)

    if not os.path.exists(args.training_log_file):
        with open(args.training_log_file, "w") as f:
            f.write("Training Config:\n")
            f.write(f"Model Hidden Layers={args.hidden_layer_size}\n")
            f.write(f"Batch Size={args.batch_size}\n")
            f.write(f"Learning Rate={args.learning_rate}\n")
            f.write(f"Max Epochs={args.max_epochs}\n")
            f.write(f"Sample Size={args.sample_size}\n")
            f.write("\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Hidden layers: {args.hidden_layer_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Log file: {args.training_log_file}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model export path: {args.model_export_path}")

    model = Model(args.hidden_layer_size).to(device)

    if args.existing_pth_file:
        model.load_state_dict(
            torch.load(args.existing_pth_file, map_location=device, weights_only=True),
            strict=False,
        )
        print(f"Loaded existing model file {args.existing_pth_file}")

    if args.fine_tuning:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    else:
        optimizer = Ranger(model.parameters(), lr=args.learning_rate)

    if not os.path.isfile(args.validation_file):
        print(
            f"Error: validation file '{args.validation_file}' does not exist or is not a file."
        )
        sys.exit(1)

    validation_dataset = MmapFileDataset(args.validation_file)

    validation_loss = evaluate_validation(
        model=model,
        criterion=criterion,
        device=device,
        batch_size=args.batch_size,
        validation_dataset=validation_dataset,
    )

    print(f"Initial validation loss: {validation_loss:.6f}")

    file_list = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)]

    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=args.total_steps
    )

    model.train()

    start_time = datetime.now()

    trained_files_count = {file: 0 for file in file_list}
    total_samples_trained = 0
    total_batch_count = 0
    last_save_time = None

    for epoch in range(args.max_epochs):
        epoch_start_time = datetime.now()
        selected_files = random.sample(file_list, args.sample_size)
        num_selected_files = len(selected_files)

        for file_name in selected_files:
            trained_files_count[file_name] += 1

        epoch_loss = 0.0
        epoch_samples_trained = 0
        batch_count = 0

        for file_idx, file_path in enumerate(selected_files):
            dataset = MmapFileDataset(file_path)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=1,
                persistent_workers=False,
            )

            file_start_time = datetime.now()
            batch_start_time = datetime.now()
            file_samples_trained = 0
            total_batches = len(dataloader)
            try:
                for batch_idx, (
                    batch_features,
                    batch_targets,
                ) in enumerate(dataloader):
                    batch_features = batch_features.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_features)

                    batch_targets = batch_targets.to(device)
                    loss = criterion(outputs.squeeze(), batch_targets)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    batch_loss = loss.item()

                    epoch_loss += batch_loss

                    samples_count = len(batch_features)
                    epoch_samples_trained += samples_count
                    total_samples_trained += samples_count
                    file_samples_trained += samples_count

                    batch_time = (datetime.now() - batch_start_time).total_seconds()
                    batch_count += 1
                    total_batch_count += 1

                    if args.save_cycle > 0 and total_batch_count % args.save_cycle == 0:
                        torch.save(
                            model.state_dict(),
                            f"{args.model_export_path}/{model.pub_name()}.pth",
                        )
                        last_save_time = datetime.now()

                    avg_epoch_loss_so_far = (
                        epoch_loss / max(batch_count, 1) if batch_count > 0 else 0.0
                    )
                    total_training_time = (datetime.now() - start_time).total_seconds()
                    current_lr = optimizer.param_groups[0]["lr"]
                    samples_per_sec = (
                        total_samples_trained / total_training_time
                        if total_training_time > 0
                        else 0
                    )
                    last_save_str = (
                        f"Last Saved: {last_save_time.strftime('%H:%M:%S')}"
                        if last_save_time
                        else "Last Saved: --"
                    )
                    print_progress_bar(
                        batch_idx + 1,
                        total_batches,
                        prefix=f"File {file_idx + 1}/{num_selected_files}:",
                        suffix=f"Batch {batch_idx + 1}/{total_batches} | Avg Loss: {avg_epoch_loss_so_far:.4f} | LR: {current_lr:.6f} | Samples: {(total_samples_trained / MILLION_BASE):.2f}M ({(samples_per_sec / MILLION_BASE):.2f}M/s) | Time: {total_training_time / SECONDS_PER_HOUR:.2f}h | {last_save_str}",
                        length=30,
                    )

                    batch_start_time = datetime.now()
            finally:
                print("\r" + " " * 100 + "\r", end="")
                file_time = (datetime.now() - file_start_time).total_seconds()
                total_training_time = (datetime.now() - start_time).total_seconds()
                grad_norm, _ = check_gradient_norm(model)
                samples_per_second = (
                    file_samples_trained / file_time if file_time > 0 else 0
                )

                avg_epoch_loss_so_far = (
                    epoch_loss / max(batch_count, 1) if batch_count > 0 else 0.0
                )

                print(
                    f"{bold('Epoch')} {epoch}/{args.max_epochs} "
                    f"{bold('File')} {file_idx + 1}/{num_selected_files} "
                    f"{bold('Samples')}: {(total_samples_trained / MILLION_BASE):.2f}M "
                    f"{bold('Avg Loss')}: {avg_epoch_loss_so_far:.4f} "
                    f"{bold('Grad Norm')}: {grad_norm:.4f} "
                    f"{bold('Samples/sec')}: {(samples_per_second / MILLION_BASE):.2f}M "
                    f"{bold('File Time')}: {file_time:.1f}s "
                    f"{bold('Total Time')}: {total_training_time / SECONDS_PER_HOUR:.2f}h "
                    f"{bold('LR')}: {optimizer.param_groups[0]['lr']:.6f}"
                )

                dataset.close()
                del dataloader
                del dataset

        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        total_time = (datetime.now() - start_time).total_seconds()

        samples_per_sec_epoch = (
            epoch_samples_trained / epoch_time if epoch_time > 0 else 0
        )

        validation_loss = evaluate_validation(
            model=model,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            validation_dataset=validation_dataset,
        )

        with open(args.training_log_file, "a") as f:
            f.write("--- Epoch Summary ---\n")
            f.write(f"Epoch={epoch}/{args.max_epochs}\n")
            f.write(f"Avg Loss={avg_epoch_loss:.6f}\n")
            f.write(f"Validation Loss={validation_loss:.6f}\n")
            f.write(f"Epoch Time (h)={epoch_time / SECONDS_PER_HOUR:.3f}\n")
            f.write(f"Total Time (h)={total_time / SECONDS_PER_HOUR:.3f}\n")
            f.write(f"Avg Samples/sec={samples_per_sec_epoch:.0f}\n")
            f.write(f"Learning Rate={optimizer.param_groups[0]['lr']:.6f}\n\n")

        print(
            f"--- Epoch {epoch}/{args.max_epochs} Completed --- \n"
            f"{bold('Loss')}: {avg_epoch_loss:.4f}\n"
            f"{bold('Validation Loss')}: {validation_loss:.4f}\n"
            f"{bold('Epoch Time')}: {epoch_time / SECONDS_PER_HOUR:.2f}h\n"
            f"{bold('Total Time')}: {total_time / SECONDS_PER_HOUR:.2f}h\n"
            f"{bold('Avg Samples/sec')}: {samples_per_sec_epoch:.0f}\n"
            f"{bold('LR')}: {optimizer.param_groups[0]['lr']:.6f}\n\n"
        )

        torch.save(
            model.state_dict(), f"{args.model_export_path}/{model.pub_name()}.pth"
        )
        print(
            f"[CHECKPOINT] Saved model to {args.model_export_path}/{model.pub_name()}.pth"
        )

    validation_dataset.close()
    print("[COMPLETE] Saving final model...")

    torch.save(model.state_dict(), f"{args.model_export_path}/{model.pub_name()}.pth")
    print(f"[FINAL] {args.model_export_path}/{model.pub_name()}.pth")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_training_time = (datetime.now() - start_time).total_seconds()
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = total_training_time % 60

    print(f"Total samples trained: {total_samples_trained:,}")
    print(f"Total training time: {hours}h {minutes}m {seconds:.1f}s")
    print(
        f"Average samples per second: {total_samples_trained / total_training_time:.1f}"
    )
