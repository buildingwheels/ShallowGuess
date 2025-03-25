#!/bin/zsh

TARGET_FEATURES=(
    "sse:+sse3,+ssse3,+sse4.1,+sse4.2,+bmi1,+bmi2"
    "avx2:+avx2,+bmi1,+bmi2"
)

FINAL_OUTPUT_DIR="binaries"
mkdir -p "$FINAL_OUTPUT_DIR"

for features in "${TARGET_FEATURES[@]}"; do
    name="${features%%:*}"
    flags="${features#*:}"
    
    echo "Building for Linux with features: $name"
    
    OUTPUT_DIR="target/linux/$name"
    mkdir -p "$OUTPUT_DIR"
    
    RUSTFLAGS="-C target-feature=$flags" cargo build --release \
        --target x86_64-unknown-linux-gnu \
        --target-dir "$OUTPUT_DIR"

    BINARY_NAME=$(basename $(cargo metadata --format-version 1 | jq -r '.packages[0].name'))
    mv "$OUTPUT_DIR/x86_64-unknown-linux-gnu/release/$BINARY_NAME" \
        "${FINAL_OUTPUT_DIR}/${BINARY_NAME}_linux_amd64_${name}"
done

for features in "${TARGET_FEATURES[@]}"; do
    name="${features%%:*}"
    flags="${features#*:}"
    
    echo "Building for Windows with features: $name"
    
    OUTPUT_DIR="target/windows/$name"
    mkdir -p "$OUTPUT_DIR"
    
    RUSTFLAGS="-C target-feature=$flags" cargo build --release \
        --target x86_64-pc-windows-gnu \
        --target-dir "$OUTPUT_DIR"
    
    BINARY_NAME=$(basename $(cargo metadata --format-version 1 | jq -r '.packages[0].name'))
    mv "$OUTPUT_DIR/x86_64-pc-windows-gnu/release/$BINARY_NAME.exe" \
        "${FINAL_OUTPUT_DIR}/${BINARY_NAME}_win_amd64_${name}.exe"
done