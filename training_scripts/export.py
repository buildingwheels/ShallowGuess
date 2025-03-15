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

for name, module in quantized_model.named_modules():
    if isinstance(module, quantization.QuantStub):
        print(
            f"[Warning: not expected] Input scaling factor: {module.scale}, zero-point: {module.zero_point}"
        )
    elif isinstance(module, nn.quantized.Linear):
        assert(module.weight().q_zero_point() == 0.)

with torch.no_grad():
    fc1_weight = quantized_model.fc1.weight()
    fc1_bias = quantized_model.fc1.bias().dequantize()

    fc1_weight_int = fc1_weight.int_repr()
    scaling_factor = fc1_weight.q_scale()

    fc2_weight = quantized_model.fc2.weight().dequantize()
    fc2_bias = quantized_model.fc2.bias().dequantize()

    fc1_weight_int = np.atleast_1d(fc1_weight_int.numpy().flatten())
    fc1_bias = np.atleast_1d(fc1_bias.numpy().flatten())
    fc2_weight = np.atleast_1d(fc2_weight.numpy().flatten())
    fc2_bias = np.atleast_1d(fc2_bias.numpy().flatten())

    flattened_weights_biases = np.concatenate([
        fc1_weight_int,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        np.array([scaling_factor]),
    ])


with open(export_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(flattened_weights_biases)

print(f"Quantized weights and biases saved to {export_file}")
