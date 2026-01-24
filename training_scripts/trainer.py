import argparse
import os
import random
import signal
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from dataset_utils import mmap_collate_fn
from torch.utils.data import DataLoader

HUNDRED_BASE = 100.0
MILLION_BASE = 1000000.0
SECONDS_PER_HOUR = 3600


training_interrupted = False


def bold(text):
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


def evaluate_validation(
    model,
    criterion,
    device,
    model_type,
    batch_size,
    validation_dataset,
):
    dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=False,
        collate_fn=mmap_collate_fn,
    )

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_features, batch_targets in dataloader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            if model_type == "coach":
                loss = criterion(model(batch_features), batch_targets)
            else:
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_targets)

            total_loss += loss.item() * len(batch_features)
            total_samples += len(batch_features)

    model.train(was_training)

    del dataloader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=["coach", "player"])
    parser.add_argument("hidden_layer_size", type=int)
    parser.add_argument("data_dir")
    parser.add_argument("validation_file")
    parser.add_argument("model_export_path")
    parser.add_argument("max_epochs", type=int)
    parser.add_argument("sample_size", type=int)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--print_cycle", type=int, default=10)
    parser.add_argument("--export_cycle", type=int, default=1000)
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
        "--alpha",
        type=float,
        default=0.7,
        help="Mixing weight for coach vs ground truth loss (player model only)",
    )

    args = parser.parse_args()

    from coach_model import CoachModel
    from player_model import PlayerModel

    if args.model_type == "coach":
        from coach_dataset import MmapFileDataset

        Model = CoachModel
        criterion = nn.CrossEntropyLoss()
        log_suffix = "_coach"
    else:
        from player_dataset import MmapFileDataset

        Model = PlayerModel
        criterion = nn.MSELoss()
        log_suffix = "_player"

    from dataset_utils import mmap_collate_fn

    if args.training_log_file is None:
        args.training_log_file = (
            f"temp/training_log{log_suffix}_{args.hidden_layer_size}.log"
        )

    os.makedirs(os.path.dirname(args.training_log_file), exist_ok=True)

    if not os.path.exists(args.training_log_file):
        with open(args.training_log_file, "w") as f:
            f.write("Training Config:\n")
            f.write(f"Model Type={args.model_type}\n")
            f.write(f"Model Hidden Layers={args.hidden_layer_size}\n")
            f.write(f"Batch Size={args.batch_size}\n")
            f.write(f"Learning Rate={args.learning_rate}\n")
            f.write(f"Max Epochs={args.max_epochs}\n")
            f.write(f"Sample Size={args.sample_size}\n")
            if args.model_type == "player":
                f.write(f"Alpha (coach weight)={args.alpha}\n")
            f.write("\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Hidden layers: {args.hidden_layer_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    if args.model_type == "player":
        print(f"Alpha (coach weight): {args.alpha}")
    print(f"Log file: {args.training_log_file}")

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
        optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    if not os.path.isfile(args.validation_file):
        print(
            f"Error: validation file '{args.validation_file}' does not exist or is not a file."
        )
        sys.exit(1)

    if args.model_type == "coach":
        from coach_dataset import MmapFileDataset as ValidationDataset
    else:
        from player_validation_dataset import (
            PlayerValidationDataset as ValidationDataset,
        )
    validation_dataset = ValidationDataset(args.validation_file)

    validation_loss = evaluate_validation(
        model=model,
        criterion=criterion,
        device=device,
        model_type=args.model_type,
        batch_size=args.batch_size,
        validation_dataset=validation_dataset,
    )
    print(f"{bold('Initial Validation Loss')}: {validation_loss:.4f}")
    with open(args.training_log_file, "a") as f:
        f.write("--- Initial Validation ---\n")
        f.write(f"Validation Loss={validation_loss:.6f}\n\n")

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
        epoch_samples_trained = 0
        batch_count = 0

        for file_idx, file_path in enumerate(selected_files):
            if training_interrupted:
                break

            dataset = MmapFileDataset(file_path)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                persistent_workers=False,
                collate_fn=mmap_collate_fn,
            )

            file_start_time = datetime.now()
            try:
                for batch_idx, (
                    batch_features,
                    batch_targets,
                ) in enumerate(dataloader):
                    if training_interrupted:
                        break

                    batch_start_time = datetime.now()
                    batch_features = batch_features.to(device)

                    batch_weights = None

                    optimizer.zero_grad()
                    outputs = model(batch_features)

                    if args.model_type == "coach":
                        batch_targets = batch_targets.to(device)
                        loss = criterion(outputs, batch_targets)
                    else:
                        if not isinstance(batch_targets, tuple):
                            raise ValueError(
                                f"Player model expects tuple (win_probs, original_results), got {type(batch_targets)}"
                            )
                        batch_win_probs, batch_original_results = batch_targets
                        batch_win_probs = batch_win_probs.to(device)
                        batch_original_results = batch_original_results.to(device)

                        coach_loss = criterion(outputs.squeeze(), batch_win_probs)
                        ground_truth_loss = criterion(
                            outputs.squeeze(), batch_original_results
                        )
                        loss = (
                            args.alpha * coach_loss
                            + (1 - args.alpha) * ground_truth_loss
                        )

                    loss.backward()
                    optimizer.step()

                    batch_loss = loss.item()

                    epoch_loss += batch_loss

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
                            f"{bold('Avg Loss')}: {(epoch_loss / max(batch_count, 1)):.4f} "
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
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        total_time = (datetime.now() - start_time).total_seconds()

        validation_loss = evaluate_validation(
            model=model,
            criterion=criterion,
            device=device,
            model_type=args.model_type,
            batch_size=args.batch_size,
            validation_dataset=validation_dataset,
        )

        scheduler.step(validation_loss)

        samples_per_sec_epoch = (
            epoch_samples_trained / epoch_time if epoch_time > 0 else 0
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

    validation_dataset.close()
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

    print(f"Total samples trained: {total_samples_trained:,}")
    print(f"Total training time: {hours}h {minutes}m {seconds:.1f}s")
    print(
        f"Average samples per second: {total_samples_trained / total_training_time:.1f}"
    )

    if training_interrupted:
        print("[DONE] Training stopped successfully.")
        sys.exit(0)
