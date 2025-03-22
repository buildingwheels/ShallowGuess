import os
import random
import sys
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.ao.quantization as quantization

from model import ShallowGuessNetwork


class FileDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, "r") as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        values = line.split(",")

        decompressed_features = []
        for value in values[:-1]:
            if value.isdigit():
                decompressed_features.extend([0.] * int(value))
            else:
                decompressed_features.append(1.)

        features = torch.tensor(decompressed_features, dtype=torch.float32)
        result = torch.tensor([float(values[-1])], dtype=torch.float32)

        return features, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hidden_layer_size", type=int)
    parser.add_argument("data_dir")
    parser.add_argument("model_export_path")
    parser.add_argument("max_epochs", type=int)
    parser.add_argument("sample_size", type=int)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--print_cycle", type=int, default=10)
    parser.add_argument("--export_cycle", type=int, default=1000)
    parser.add_argument("--existing_pth_file", default=None)
    args = parser.parse_args()

    device = torch.device('cuda')

    model = ShallowGuessNetwork(args.hidden_layer_size).to(device)
    model.qconfig = quantization.default_qconfig
    quantization.prepare_qat(model, inplace=True)

    if args.existing_pth_file:
        model.load_state_dict(torch.load(args.existing_pth_file, weights_only=True), strict=False)
        print(f"Loaded existing model file {args.existing_pth_file}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    file_list = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)]

    model.train()

    start_time = datetime.now()

    trained_files_count = {file: 0 for file in file_list}
    total_samples_trained = 0

    for epoch in range(args.max_epochs):
        selected_files = random.sample(file_list, args.sample_size)
        num_selected_files = len(selected_files)

        for file_name in selected_files:
            trained_files_count[file_name] += 1

        epoch_loss = 0.0
        epoch_samples_trained = 0
        batch_count = 0

        for file_idx, file_path in enumerate(selected_files):
            dataset = FileDataset(file_path)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            for batch_idx, (batch_features, batch_results) in enumerate(dataloader):
                batch_features = batch_features.to(device)
                batch_results = batch_results.to(device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_results)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss

                samples_count = len(batch_features)
                epoch_samples_trained += samples_count
                total_samples_trained += samples_count

                progress = epoch_samples_trained / (len(dataset) * num_selected_files) * 100.
                batch_count += 1

                if batch_idx % args.print_cycle == 0:
                    print(
                        f"Trained [{(total_samples_trained / 1000000.):.2f}M Positions], "
                        f"Epoch [{epoch}/{args.max_epochs}], "
                        f"File [{file_idx + 1}/{num_selected_files}], "
                        f"Batch [{batch_idx + 1}/{len(dataloader)}], "
                        f"Progress [{progress:.2f}%], "
                        f"Batch Loss [{batch_loss:.3f}], "
                        f"Epoch Loss [{(epoch_loss / batch_count):.5f}], "
                        f"Training time: {datetime.now() - start_time}"
                    )

                if batch_idx % args.export_cycle == 0:
                    model.eval()
                    torch.save(model.state_dict(), f"{args.model_export_path}/{model.pub_name()}.pth")
                    model.train()

        avg_epoch_loss = epoch_loss / batch_count

        print(
            f"Epoch [{epoch}/{args.max_epochs}] completed with error [{avg_epoch_loss:.4f}]"
        )
