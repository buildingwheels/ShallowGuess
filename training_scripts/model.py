import torch
import torch.nn as nn

FIXED_INPUT_LAYER_SIZE = 768
FIXED_OUTPUT_LAYER_SIZE = 3

class ShallowGuessNetwork(nn.Module):
    def __init__(self, hidden_layer_size):
        super(ShallowGuessNetwork, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.fc1 = nn.Linear(FIXED_INPUT_LAYER_SIZE, hidden_layer_size, dtype=torch.float32)
        self.fc2 = nn.Linear(hidden_layer_size, FIXED_OUTPUT_LAYER_SIZE, dtype=torch.float32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def pub_name(self):
        return f"{self.hidden_layer_size}"
