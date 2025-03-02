
![logo](https://github.com/user-attachments/assets/ac8030d7-21a3-47c5-afa8-9ecc00883eb2)

# Shallow Guess

A chess engine powered by neural networks. Still in early preview, work in progress.

## Board Representation

- Basic/Naive Bitboards (No Magic)

## Move Generation
- Pre-calculated bit masks
- Runtime ray attack generation with LSB/HSB operations
- Phased move generation

## Search Heuristics

- Principal Variation Search
- Iterative Deepening
- Internal Iterative Deepening
- Aspiration Window
- Quiescence Search
- Static Exchange Evaluation (SEE)
- Late Move Reduction
- Counter Move Heuristics
- History Heuristics
- Killer Heuristics
- Null Move Pruning
- Futility Pruning

## Hashing & Transposition
- Zobrist Hashing
- Always-replacing Transposition Table

## Evaluation
- Neural network with single hidden layer
- Partially quantized (only the 1st fully connected layer which is used for incremental updates)

## How to Use
There are 3 trained models under the `resources/models/` folder:
- `1L-32.pth` (used for experiments, around 2000 ELO)
- `1L-128.pth` (default, around 2500 ELO)
- `1L-256.pth` (not ready, training in progress)

The associated weights have been exported into the `resources/weights/` folder with `scripts/export.py`:
- `1L-32.weights`
- `1L-128.weights`
- `1L-256.weights`

You can modify `build.rs` to load one of the three trained models:
```rust
fn main() {
    load_weights(768, <hidden_layer_size>);
}
```
You may change the value of <hidden_layer_size> to `32`, `128`, `256`, or other values if you have trained your own model with different sizes.

Then run:
```bash
cargo build --release
```

You can load the built binary `target/release/shallow_guess` in any UCI-compatible GUI.

## How to Train Your Own Model
### Data Preparation
1. Create a FEN file from PGN file with [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/):
```bash
pgn-extract -Wfen <your-pgn-file> > raw-fen.txt
```

2. Run the following command to generate training data:
```bash
cargo run --bin gen_training_set --release raw-fen.txt pre-process-fen.txt result.txt <skip-count> <batch-size>
```

### Training
**Note**: you need to install `pytorch`.

Use scripts provided in the `scripts` folder:
1. Run the following command to start training:
```bash
python scripts/trainer.py <hidden-layer-size> <training-data-folder> <output-folder> <max-epochs> <(optional) existing-model-file>
```

2. After training is completed, run the following command to export weights:
```bash
python scripts/export.py <hidden-layer-size> <model-file> <output-weight-file>
```


## Credits
### PGN Extract
[pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/) was used for extracting training positions from PGN files.

### CCRL (Computer Chess Rating Lists)
Big part of the training data were generated based on history games found on CCRL website.

### Lichess Open Database & Lichess Elite Database
Part of the training data were generated from games randomly selected from these websites:
- [Lichess Open Database](https://database.lichess.org/)
- [Lichess Elite Database](https://database.nikonoel.fr/)

### TCEC (Top Chess Engine Championship)
Part of the training data were generated from history [TCEC](https://tcec-chess.com/) tournament games.
