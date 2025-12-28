# Shallow Guess

A strong UCI-compatible chess engine powered by a neural network **trained solely on game results**.

## Features

### Board Representation and Move Generation
- **Naive Bitboards**
- **Phased Move Generation**

### Search Algorithm
- **Principal Variation Search**
- **Iterative Deepening**
- **Alpha-Beta Pruning**
- **Aspiration Windows**
- **Quiescence Search**
- **Static Exchange Evaluation (SEE)**
- **History Heuristic**
- **Killer Heuristic**
- **Counter Move Heuristic**
- **Follow-up Move Heuristic**
- **Late Move Reductions**
- **Null Move Pruning**
- **Zobrist Hashing**
- **Two-tier Transposition Table**

### Evaluation
- **Partially Quantized Neural Network**
- **Configurable Architecture**
- **Training on Game Results Only**
- **SIMD Optimizations** (SSE, AVX2)

#### Network Architecture
```
Input Layer: 768 neurons (12 piece types × 64 squares)
        ↓
Hidden Layer: N neurons (configurable, e.g., 512)
        ↓
Output Layer: 3 neurons (Loss/Draw/Win probabilities)
```

The network uses **dynamic quantization** (int8) for the input-to-hidden layer weights to optimize inference performance.

#### Available Models

The engine comes with three pre-trained models of different hidden layer sizes:

| Model | Hidden Layer Size | Model File | Weights File |
|-------|-------------------|------------|--------------|
| **512** (default) | 512 | `resources/models/512.pth` | `resources/weights/512.weights` |
| **256** | 256 | `resources/models/256.pth` | `resources/weights/256.weights` |
| **1024** | 1024 | `resources/models/1024.pth` | `resources/weights/1024.weights` |

**Note:** The 512 model is the default for release builds.

#### Switching Models

To use a different model, edit `config/network.cfg` and set the `hidden_layer_size`:

```bash
hidden_layer_size=<256/512/1024>
```

Then rebuild the engine:

```bash
cargo build --release
```

## Training Model

The neural network is trained on chess game results using a multi-step pipeline that processes PGN data into training-ready format.

For details, reference to **[TrainingGuide.md](TrainingGuide.md)**.

## Utility Programs

### Engine Parameter Testing
The `param_test` utility tests engine parameters against EPD test suites.

```bash
cargo run --bin param_test [epd_file] [search_time_secs]
```

### Zobrist Key Generation
The `zobrist_key_gen` utility generates optimal hash tables by testing multiple random seeds to minimize collisions.

```bash
cargo run --bin zobrist_key_gen [fen_file_path] [max_seeds_count]
```

## Build

### Pre-compiled Binaries
Pre-compiled binaries have been removed since 1.0 due to complexity in supporting multiple CPU features. Please compile directly from source code.

### Compile from Source

#### Prerequisites
- **Rust Nightly** - This project requires Rust nightly for SIMD support. Install from [rustup.rs](https://rustup.rs/)
- **Git** - To clone the repository
- **Python 3.10+ & PyTorch** - Only needed for training new models
- **jq** - For parsing JSON metadata (required for build script)

**Switch to Rust Nightly:**
```bash
rustup install nightly
rustup default nightly
# Or override for just this project:
rustup override set nightly
```

#### Build Steps

##### Quick Build (Single Binary)
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/shallow-guess.git
   cd shallow-guess
   ```

2. **Verify model files:**
   - Trained models are in `resources/models/`
   - Exported weights are in `resources/weights/`
   - Configure model in `config/network.cfg`

3. **Build the engine:**
   ```bash
    export RUSTFLAGS="-C target-cpu=native"
    cargo build --release
   ```

4. **Run the engine:**
    ```bash
     ./target/release/shallow_guess
    ```

## Optimized CPU Target Features

This engine includes SIMD optimizations for various CPU instruction sets. The build system automatically detects and uses the best available features based on your CPU.

### Supported Instruction Sets

| Feature | SIMD Types | SIMD Lane Width |
|---------|------------|-----------------|
| **AVX-512F** | `f32x16`, `i16x32` | 16 |
| **AVX2/AVX** | `f32x8`, `i16x16` | 8 |
| **SSE4.1/SSE2** | `f32x4`, `i16x8` | 4 |
| **Default** | `f32x4`, `i16x8` | 4 |

### Building with Specific Features

The `target-cpu=native` flag enables all available features for your CPU:

```bash
# Build with native CPU optimizations (recommended)
export RUSTFLAGS="-C target-cpu=native"
cargo build --release
```

To build for specific instruction sets, use:

```bash
# AVX-512F only
export RUSTFLAGS="-C target-feature=+avx512f"
cargo build --release

# AVX2 only
export RUSTFLAGS="-C target-feature=+avx2"
cargo build --release

# SSE4.1 only
export RUSTFLAGS="-C target-feature=+sse4.1"
cargo build --release

# Build for x86-64 baseline (SSE2)
export RUSTFLAGS="-C target-cpu=x86-64"
cargo build --release
```

### Checking Available Features

To check which CPU features are available on your system:

```bash
# Linux
lscpu | grep "Flags"

# macOS
sysctl -n machdep.cpu.features

# Or check during build
cargo build --release 2>&1 | grep "target-feature"
```

### Architecture Support

- **x86_64**: Full support for AVX-512F, AVX2/AVX, SSE4.1/SSE2
- **aarch64**: Support for ARM NEON (f32x4, i16x8)
- **Other**: Fallback to 4-lane SIMD

## Acknowledgments

### PGN Extract
[pgn-extract](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/) was used for extracting training positions from PGN files.

### CCRL (Computer Chess Rating Lists)
All training data for the latest release version was generated from historical 40/15 games downloaded from the CCRL website.

### TCEC (Top Chess Engine Championship)
Training data for previous test versions was generated from [TCEC](https://tcec-chess.com/) tournament games.

### Chacha20 by Daniel J. Bernstein

Pseudo random number generator is a implementation of Chacha20 algorithm authored by Daniel J. Bernstein.
