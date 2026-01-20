cd "$(dirname "$0")/.."

sed -i "s/hidden_layer_size=[0-9]\+/hidden_layer_size=$1/" config/network.cfg

MODEL_FILE="resources/models/Player-$1.pth"

echo "Using model file: $MODEL_FILE"
echo "Exporting weights..."
python training_scripts/export_player_model.py $1 "$MODEL_FILE" resources/raw_weights/$1.raw_weights

echo "Quantizing weights..."
cargo run --release --bin quantize_weights $1 resources/raw_weights/$1.raw_weights resources/quantized_weights/$1.quantized_weights

echo "Done exporting and quantizing weights"

export RUSTFLAGS="-C target-cpu=native"
cargo build --release

echo "Updated network weights used in binary"
