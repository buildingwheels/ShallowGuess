import argparse
import os
from pathlib import Path

import numpy as np
import torch


def parse_line_to_features(line):
    parts = line.split(",")

    if len(parts) != 2:
        raise ValueError(
            f"Invalid line format: expected 2 columns, got {len(parts)}: {line}"
        )

    features_str = parts[0]
    weighted_game_result = float(parts[1])

    tokens = features_str.split("X")
    num_segments = len(tokens)
    features = np.zeros(768, dtype=np.float32)
    pos = 0

    for i, token in enumerate(tokens):
        if not token:
            zero_count = 0
        else:
            zero_count = int(token)

        if i < num_segments - 1:
            end_pos = pos + zero_count + 1
            if end_pos > 768:
                end_pos = 768
            features[pos:end_pos] = 0.0
            if pos + zero_count < 768:
                features[pos + zero_count] = 1.0
            pos = end_pos
        else:
            end_pos = pos + zero_count
            if end_pos > 768:
                end_pos = 768
            features[pos:end_pos] = 0.0
            pos = end_pos

    return features, weighted_game_result


def convert_file(input_path, output_path, chunk_size):
    print(f"Converting: {input_path} -> {output_path}")

    features_list = []
    labels_list = []
    total_samples = 0
    chunk_idx = 0

    with open(input_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                features, label = parse_line_to_features(line)
                features_list.append(features)
                labels_list.append(label)
                total_samples += 1

                if len(features_list) >= chunk_size:
                    features_tensor = np.stack(features_list, axis=0)
                    labels_tensor = np.array(labels_list, dtype=np.float32)

                    chunk_file = output_path.replace(".pt", f"_chunk_{chunk_idx}.pt")
                    torch.save(
                        {
                            "features": torch.from_numpy(features_tensor).to(torch.float16),
                            "labels": torch.from_numpy(labels_tensor).to(torch.float16),
                        },
                        chunk_file,
                    )
                    print(f"Saved chunk {chunk_idx}: {len(features_list)} samples")
                    chunk_idx += 1
                    features_list = []
                    labels_list = []

            except Exception as e:
                print(f"Warning: skipping line {line_num}: {e}")
                continue

    if features_list:
        features_tensor = np.stack(features_list, axis=0)
        labels_tensor = np.array(labels_list, dtype=np.float32)

        if chunk_idx == 0:
            torch.save(
                {
                    "features": torch.from_numpy(features_tensor).to(torch.float16),
                    "labels": torch.from_numpy(labels_tensor).to(torch.float16),
                },
                output_path,
            )
            print(f"  Saved: {total_samples} samples -> {output_path}")
        else:
            chunk_file = output_path.replace(".pt", f"_chunk_{chunk_idx}.pt")
            torch.save(
                {
                    "features": torch.from_numpy(features_tensor).to(torch.float16),
                    "labels": torch.from_numpy(labels_tensor).to(torch.float16),
                },
                chunk_file,
            )
            print(f"Saved chunk {chunk_idx}: {len(features_list)} samples")
            print(f"Total: {total_samples} samples in {chunk_idx + 1} chunk files")
    elif chunk_idx == 0:
        print(f"Warning: No samples found in {input_path}")

    return total_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text training data to .pt format")
    parser.add_argument("input_path", help="Input text file or directory")
    parser.add_argument("output_path", help="Output .pt file or directory")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="Samples per chunk file")

    args = parser.parse_args()

    if os.path.isdir(args.input_path):
        os.makedirs(args.output_path, exist_ok=True)

        total = 0
        files = [f for f in os.listdir(args.input_path)]

        for filename in files:
            input_file = os.path.join(args.input_path, filename)
            output_file = os.path.join(args.output_path, filename + ".pt")
            total += convert_file(input_file, output_file, args.chunk_size)

        print(f"\nConversion complete: {total} total samples")
    else:
        convert_file(args.input_path, args.output_path, args.chunk_size)
