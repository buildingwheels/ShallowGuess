import torch
import torch.nn as nn

FIXED_INPUT_LAYER_SIZE = 768
FIXED_OUTPUT_LAYER_SIZE = 1


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight_q = self._quantize_with_ste(self.weight)

        return nn.functional.linear(x, weight_q, self.bias)

    def _quantize_with_ste(self, tensor, bits=8):
        if tensor is None:
            return None

        qmax = 2**(bits-1) - 1
        qmin = -2**(bits-1)

        max_abs = torch.max(torch.abs(tensor))
        scale = max_abs / qmax + 1e-6

        with torch.no_grad():
            quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)

        dequantized = quantized * scale

        if tensor.requires_grad:
            dequantized = tensor + (dequantized - tensor).detach()

        return dequantized


class PlayerModel(nn.Module):
    def __init__(self, hidden_layer_size):
        super(PlayerModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.fc1 = QuantizedLinear(FIXED_INPUT_LAYER_SIZE, hidden_layer_size)
        self.relu = nn.LeakyReLU()

        self.fc2 = nn.Linear(
            hidden_layer_size, FIXED_OUTPUT_LAYER_SIZE, dtype=torch.float32
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

    def pub_name(self):
        return f"Player-{self.hidden_layer_size}"
