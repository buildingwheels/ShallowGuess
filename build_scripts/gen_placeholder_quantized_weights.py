import sys
from pathlib import Path

INPUT_LAYER_SIZE = 768


def generate_placeholder_quantized_weights(hidden_layer_size, output_path):
    fc1_weight_count = INPUT_LAYER_SIZE * hidden_layer_size
    fc1_bias_count = hidden_layer_size
    fc2_weight_count = hidden_layer_size

    total_elements = fc1_weight_count + fc1_bias_count + fc2_weight_count + 2

    with open(output_path, "w") as f:
        for _ in range(fc1_weight_count):
            f.write("0,")

        for _ in range(fc1_bias_count):
            f.write("0.0,")

        for _ in range(fc2_weight_count):
            f.write("0.0,")

        f.write("0.0,")

        f.write("0.0\n")

    print(f"Generated placeholder quantized weights: {output_path}")
    print(f"  fc1_weights: {fc1_weight_count} (int8)")
    print(f"  fc1_bias: {fc1_bias_count} (f32)")
    print(f"  fc2_weights: {fc2_weight_count} (f32)")
    print("  fc2_bias: 1 (f32)")
    print("  scale: 1 (f32)")
    print(f"  Total elements: {total_elements}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <hidden_layer_size>")
        sys.exit(1)

    try:
        hidden_layer_size = int(sys.argv[1])
    except ValueError:
        print("Error: hidden_layer_size must be an integer")
        sys.exit(1)

    output_dir = Path("resources/quantized_weights")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{hidden_layer_size}.quantized_weights"

    generate_placeholder_quantized_weights(hidden_layer_size, output_path)
