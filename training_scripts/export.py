import argparse
import csv
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quantization
from model import ShallowGuessNetwork


def load_model(hidden_layer_size, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ShallowGuessNetwork(hidden_layer_size)
    model.load_state_dict(
        torch.load(model_file, map_location=device, weights_only=True)
    )
    model.eval()

    quantized_model = quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized_model


def extract_weights_biases(quantized_model):
    with torch.no_grad():
        fc1_weight = quantized_model.fc1.weight()
        fc1_bias = quantized_model.fc1.bias().dequantize()
        fc1_weight_int = fc1_weight.int_repr()
        scaling_factor = fc1_weight.q_scale()

        fc2_weight = quantized_model.fc2.weight().dequantize()
        fc2_bias = quantized_model.fc2.bias().dequantize()

        fc1_weight_int = np.atleast_1d(fc1_weight_int.flatten())
        fc1_bias = np.atleast_1d(fc1_bias.numpy().flatten())
        fc2_weight = np.atleast_1d(fc2_weight.numpy().flatten())
        fc2_bias = np.atleast_1d(fc2_bias.numpy().flatten())

        return np.concatenate(
            [fc1_weight_int, fc1_bias, fc2_weight, fc2_bias, np.array([scaling_factor])]
        )


def save_to_csv(data, export_file):
    with open(export_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
    print(f"Dynamically quantized weights and biases saved to {export_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export dynamically quantized model weights and biases to a CSV file."
    )
    parser.add_argument(
        "hidden_layer_size", type=int, help="Size of the hidden layer in the model."
    )
    parser.add_argument("model_file", help="Path to the trained model file.")
    parser.add_argument("export_file", help="Path to the output CSV file.")
    args = parser.parse_args()

    try:
        quantized_model = load_model(args.hidden_layer_size, args.model_file)
        flattened_data = extract_weights_biases(quantized_model)
        save_to_csv(flattened_data, args.export_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
