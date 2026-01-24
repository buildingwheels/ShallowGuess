import torch
import torch.nn as nn

FIXED_INPUT_LAYER_SIZE = 768
FIXED_2ND_HIDDEN_LAYER_SIZE = 32
FIXED_OUTPUT_LAYER_SIZE = 3


class CoachModel(nn.Module):
    def __init__(self, first_hidden_layer_size):
        super(CoachModel, self).__init__()
        self.hidden_layer_size = first_hidden_layer_size

        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1, dtype=torch.float32)
        self.bn1 = nn.BatchNorm2d(32, dtype=torch.float32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, dtype=torch.float32)
        self.bn2 = nn.BatchNorm2d(64, dtype=torch.float32)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU()

        self.fc1 = nn.Linear(64 * 2 * 2, first_hidden_layer_size, dtype=torch.float32)
        self.fc2 = nn.Linear(
            first_hidden_layer_size, FIXED_2ND_HIDDEN_LAYER_SIZE, dtype=torch.float32
        )
        self.fc3 = nn.Linear(
            FIXED_2ND_HIDDEN_LAYER_SIZE, FIXED_OUTPUT_LAYER_SIZE, dtype=torch.float32
        )

    def forward(self, x):
        x = x.view(-1, 12, 8, 8)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

    def pub_name(self):
        return f"Coach-{self.hidden_layer_size}"
