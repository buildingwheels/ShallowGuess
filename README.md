# Shallow Guess

A strong UCI-compatible chess engine powered by neural networks.

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
- MVV-LVA Sorting
- Late Move Reductions
- Counter Move Heuristics
- History Heuristics
- Killer Heuristics
- Null Move Pruning
- Singular Move Extensions
- Mate Threat Detection

## Hashing & Transposition
- Zobrist Hashing
- Depth-preferred Transposition Table
- Always-replacing Transposition Table

## Evaluation
- Neural network with one hidden layer
- Partially quantized (only the 1st fully-connected layer which is used for incremental updates)

## How to Use
### Use Pre-compiled Binaries
The following binaries are pre-compiled and attached in each release:
- Linux x86-64 SSE
- Linux x86-64 AVX2
- Linux x86-64 AMD
- Windows x86-64 SSE
- Windows x86-64 AVX2
- Windows x86-64 AMD

### Compile Natively (Strongly Recommended)
The trained models are placed under the `resources/models/` folder.
The associated weights have been exported into the `resources/weights/` folder with `training_scripts/export.py`.

You can modify `config/network.cfg` to load one of the trained models.

Then run:
```bash
cargo build --release
```

The built binary can be found at `target/release/shallow_guess`.

## Train Your Own Model
### Data Preparation
1. Create a FEN file from PGN file with [pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/):
```bash
pgn-extract -Wfen <your-pgn-file> > raw-fen.txt
```

2. Run the following command to generate training data:
```bash
cargo run --bin gen_training_set --release raw-fen.txt pre-process-fen.txt result.txt <skip-position-count> <max-number-of-positions-per-game> <batch-size>
```

### Training
**Note**: you need to install `pytorch`.

Use scripts provided in the `training_scripts` folder:
1. Run the following command to start training:
```bash
python training_scripts/trainer.py <hidden-layer-size> <training-data-folder> <output-folder> <max-epochs> <sample-size> (and a few other optional parameters, see code to find our more)
```

2. After training is completed, run the following command to export weights:
```bash
python training_scripts/export.py <hidden-layer-size> <model-file> <output-weight-file>
```

## Credits
### PGN Extract
[pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/) was used for extracting training positions from PGN files.

### CCRL (Computer Chess Rating Lists)
60% of the training data were generated based on history games found on CCRL website.

### Lichess Open Database & Lichess Elite Database
40%% of the training data were generated from games randomly selected from these websites:
- [Lichess Open Database](https://database.lichess.org/)
- [Lichess Elite Database](https://database.nikonoel.fr/)

### TCEC (Top Chess Engine Championship)
The training data for the intitial experimental versions were generated from [TCEC](https://tcec-chess.com/) history tournament games. Not used in the training of release versions.
