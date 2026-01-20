import argparse
import os
import sys

import numpy as np
import torch
from player_model import PlayerModel


def load_model(hidden_layer_size, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PlayerModel(hidden_layer_size)
    model.load_state_dict(
        torch.load(model_file, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def export_weights(model, export_file):
    with torch.no_grad():
        fc1_weight = model.fc1.weight.detach().cpu().numpy().flatten()
        fc1_bias = model.fc1.bias.detach().cpu().numpy().flatten()
        fc2_weight = model.fc2.weight.detach().cpu().numpy().flatten()
        fc2_bias = model.fc2.bias.detach().cpu().numpy().flatten()

    flattened_data = np.concatenate([fc1_weight, fc1_bias, fc2_weight, fc2_bias])

    os.makedirs(os.path.dirname(export_file) or ".", exist_ok=True)

    with open(export_file, "w") as f:
        f.write(",".join([str(x) for x in flattened_data]))

    print(f"Model weights exported to {export_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export model weights and biases to a flattened CSV file."
    )
    parser.add_argument(
        "hidden_layer_size", type=int, help="Size of the hidden layer in the model."
    )
    parser.add_argument("model_file", help="Path to the trained model file.")
    parser.add_argument("export_file", help="Path to the output CSV file.")
    args = parser.parse_args()

    try:
        model = load_model(args.hidden_layer_size, args.model_file)
        export_weights(model, args.export_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
