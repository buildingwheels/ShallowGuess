import torch
import torch.nn as nn

FIXED_INPUT_LAYER_SIZE = 768
FIXED_2ND_HIDDEN_LAYER_SIZE = 32
FIXED_OUTPUT_LAYER_SIZE = 3


class CoachModel(nn.Module):
    def __init__(self, first_hidden_layer_size):
        super(CoachModel, self).__init__()
        self.hidden_layer_size = first_hidden_layer_size

        self.fc1 = nn.Linear(
            FIXED_INPUT_LAYER_SIZE, first_hidden_layer_size, dtype=torch.float32
        )
        self.fc2 = nn.Linear(
            first_hidden_layer_size, FIXED_2ND_HIDDEN_LAYER_SIZE, dtype=torch.float32
        )
        self.fc3 = nn.Linear(
            FIXED_2ND_HIDDEN_LAYER_SIZE, FIXED_OUTPUT_LAYER_SIZE, dtype=torch.float32
        )
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

    def pub_name(self):
        return f"Coach-{self.hidden_layer_size}"
