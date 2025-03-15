sed -i "s/hidden_layer_size=[0-9]\+/hidden_layer_size=$1/" config/network.cfg
python training_scripts/export.py $1 resources/models/1L-$1.pth resources/weights/1L-$1.weights
cargo build --release
