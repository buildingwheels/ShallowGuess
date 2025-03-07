import csv
import sys
import torch
import torch.nn as nn
import torch.ao.quantization as quantization

import numpy as np

from model import ShallowGuessNetwork

hidden_layer_size = int(sys.argv[1])
model_file = sys.argv[2]
export_file = sys.argv[3]

model = ShallowGuessNetwork(hidden_layer_size)
model.qconfig = quantization.default_qconfig
quantization.prepare_qat(model, inplace=True)

model.load_state_dict(torch.load(model_file, weights_only=True))
quantized_model = quantization.convert(model, inplace=False)

scaling_factors = []

for name, module in quantized_model.named_modules():
    if isinstance(module, quantization.QuantStub):
        print(
            f"[Warning: not expected] Input scaling factor: {module.scale}, zero-point: {module.zero_point}"
        )
    elif isinstance(module, nn.quantized.Linear):
        scaling_factors.append(module.weight().q_scale())
        assert(module.weight().q_zero_point() == 0.)

common_scaling_factor = min(scaling_factors)

print(f"Common scaling factor selected: {common_scaling_factor}")

with torch.no_grad():
    fc1_weight = quantized_model.fc1.weight()
    fc1_bias = quantized_model.fc1.bias()

    fc1_weight_int = torch.round(fc1_weight.int_repr() * (fc1_weight.q_scale() / common_scaling_factor)).int()
    fc1_bias_int = torch.round(fc1_bias / common_scaling_factor).int()

    fc2_weight = quantized_model.fc2.weight()
    fc2_bias = quantized_model.fc2.bias()

    fc2_weight_int = torch.round(fc2_weight.int_repr() * (fc2_weight.q_scale() / common_scaling_factor)).int()

    fc1_weight_int = np.atleast_1d(fc1_weight_int.numpy().flatten())
    fc1_bias_int = np.atleast_1d(fc1_bias_int.numpy().flatten())
    fc2_weight_int = np.atleast_1d(fc2_weight_int.numpy().flatten())
    fc2_bias = np.atleast_1d(fc2_bias.numpy().flatten())

    print(f"Max weights {max(fc1_weight_int)} {max(fc2_weight_int)}, max biases {max(fc1_bias_int)}")
    print(f"Min weights {min(fc1_weight_int)} {min(fc2_weight_int)}, min biases {min(fc1_bias_int)}")

    flattened_weights_biases = np.concatenate([
        fc1_weight_int,
        fc1_bias_int,
        fc2_weight_int,
        fc2_bias,
        np.array([common_scaling_factor]),
    ])


with open(export_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(flattened_weights_biases)

print(f"Quantized weights and biases saved to {export_file}")
