import os
import random
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.ao.quantization as quantization

torch.backends.cudnn.enabled = False

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
        features = torch.tensor([float(x) for x in values[:-1]], dtype=torch.float32)
        result = torch.tensor([float(values[-1])], dtype=torch.float32)
        return features, result


hidden_layer_size = int(sys.argv[1])
data_dir = sys.argv[2]
model_export_path = sys.argv[3]
max_epochs = int(sys.argv[4])
sample_size = int(sys.argv[5])

print(f"Training for hidden_layer_size={hidden_layer_size}, data_dir={data_dir}, model_export_path={model_export_path}, max_epochs={max_epochs}")

model = ShallowGuessNetwork(hidden_layer_size)
model.qconfig = quantization.default_qconfig
quantization.prepare_qat(model, inplace=True)


if len(sys.argv) > 6:
    existing_pth_file = sys.argv[6]
    model.load_state_dict(torch.load(existing_pth_file, weights_only=True))
    print(f"Loaded existing model {existing_pth_file}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

print(f"Starting training with file list of size {len(file_list)}")

model.train()

start_time = datetime.now()

for epoch in range(max_epochs):
    print(f"Training epoch: {epoch}")

    selected_files = random.sample(file_list, sample_size)
    num_selected_files = len(selected_files)

    epoch_loss = 0.0
    epoch_samples_trained = 0
    batch_count = 0

    for file_idx, file_path in enumerate(selected_files):
        print(f"Processing file: {file_idx + 1}/{num_selected_files}: {file_path}")
        dataset = FileDataset(file_path)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        for batch_idx, (batch_features, batch_results) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_results)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            epoch_samples_trained += len(batch_features)
            progress = epoch_samples_trained / (len(dataset) * num_selected_files) * 100.
            batch_count += 1

            if batch_idx % 100 == 0:
                print(
                    f"Trained [{epoch_samples_trained}], "
                    f"Epoch [{epoch}/{max_epochs}], "
                    f"File [{file_idx + 1}/{num_selected_files}], "
                    f"Batch [{batch_idx + 1}/{len(dataloader)}], "
                    f"Progress [{progress:.2f}%], "
                    f"Batch Loss [{batch_loss:.4f}], "
                    f"Epoch Loss [{(epoch_loss / batch_count):.4f}], "
                    f"Training time: {datetime.now() - start_time}"
                )

        model.eval()
        torch.save(model.state_dict(), f"{model_export_path}/{model.pub_name()}.pth")
        model.train()
        print(f"Saved model")

    avg_epoch_loss = epoch_loss / batch_count

    print(
        f"Epoch [{epoch}/{max_epochs}] completed with error [{avg_epoch_loss:.4f}]"
    )
