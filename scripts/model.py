import torch
import torch.nn as nn
import torch.ao.quantization as quantization


class ShallowGuessNetwork(nn.Module):
    def __init__(self, hidden_layer_size):
        super(ShallowGuessNetwork, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.fc1 = nn.Linear(768, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x

    def pub_name(self):
        return f"1L-{self.hidden_layer_size}"


def load_model(model_path):
    model = ShallowGuessNetwork()
    model.qconfig = quantization.default_qconfig
    quantization.prepare_qat(model, inplace=True)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
