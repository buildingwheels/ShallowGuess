cd "$(dirname "$0")/.."

sed -i "s/hidden_layer_size=[0-9]\+/hidden_layer_size=$1/" config/network.cfg

echo "Configured to load weights for: $1"

export RUSTFLAGS="-C target-cpu=native"
cargo build --release
