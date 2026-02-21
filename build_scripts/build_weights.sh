cd "$(dirname "$0")/.."

echo "Update configuration to use $1.."

sed -i "s/hidden_layer_size=[0-9]\+/hidden_layer_size=$1/" config/network.cfg

echo "Done updating configuration"

echo "Quantizing weights..."

cargo run --release --bin quantize_weights $1 resources/raw_weights/$1.raw_weights resources/quantized_weights/$1.quantized_weights

echo "Done quantizing weights"

export RUSTFLAGS="-C target-cpu=native"
cargo build --release

echo "Updated network weights used in binary"

