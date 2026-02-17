import argparse
import os
import sys

import numpy as np
import torch
from player_model import PlayerModel


def load_model(hidden_layer_size, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PlayerModel(hidden_layer_size)
    state_dict = torch.load(model_file, map_location=device, weights_only=True)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def export_weights(model, export_file):
    with torch.no_grad():
        fc1_w = model.fc1.weight.data
        fc1_b = model.fc1.bias.data

        fc2_w = model.fc2.weight.data
        fc2_b = model.fc2.bias.data

    fc1_w_np = fc1_w.detach().cpu().numpy().flatten()
    fc1_b_np = fc1_b.detach().cpu().numpy().flatten()
    fc2_w_np = fc2_w.detach().cpu().numpy().flatten()
    fc2_b_np = fc2_b.detach().cpu().numpy().flatten()

    flattened_data = np.concatenate([fc1_w_np, fc1_b_np, fc2_w_np, fc2_b_np])

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
