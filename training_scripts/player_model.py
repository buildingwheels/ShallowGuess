import torch
import torch.nn as nn

FIXED_INPUT_LAYER_SIZE = 768
FIXED_OUTPUT_LAYER_SIZE = 1


def fake_quantize(weight, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1

    if not weight.requires_grad:
        with torch.no_grad():
            max_abs = torch.max(torch.abs(weight))
            scale = max_abs / qmax if max_abs > 1e-6 else 1.0
            quantized = torch.clamp(
                torch.round(weight / scale), min=-qmax - 1, max=qmax
            )
            dequantized = quantized * scale
        return dequantized

    max_abs = torch.max(torch.abs(weight.detach()))
    scale = max_abs / qmax if max_abs > 1e-6 else 1.0
    quantized = torch.clamp(torch.round(weight / scale), min=-qmax - 1, max=qmax)
    dequantized = quantized * scale

    return weight + (dequantized - weight).detach()


class FakeQuantLinear(nn.Linear):
    def __init__(
        self, in_features, out_features, bias=True, dtype=torch.float32, num_bits=8
    ):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.num_bits = num_bits

    def forward(self, input):
        quantized_weight = fake_quantize(self.weight, self.num_bits)
        return nn.functional.linear(input, quantized_weight, self.bias)


class PlayerModel(nn.Module):
    def __init__(self, hidden_layer_size):
        super(PlayerModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.fc1 = FakeQuantLinear(
            FIXED_INPUT_LAYER_SIZE, hidden_layer_size, dtype=torch.float32
        )
        self.fc2 = nn.Linear(
            hidden_layer_size, FIXED_OUTPUT_LAYER_SIZE, dtype=torch.float32
        )
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

    def pub_name(self):
        return f"Player-{self.hidden_layer_size}"
