#!/bin/bash

set -e

PROJECT_NAME="shallow_guess"
TARGET_DIR="binaries"

BUILD_TYPES=("release")
TARGETS=("x86_64-unknown-linux-gnu" "x86_64-pc-windows-gnu")
SIMD_PROFILES=("sse" "avx2" "avx512")

SIMD_FLAGS_SSE="-C target-feature=+sse4.1,+sse4.2"
SIMD_FLAGS_AVX2="-C target-feature=+avx2,+fma"
SIMD_FLAGS_AVX512="-C target-feature=+avx512f,+avx512dq,+avx512vl,+avx512bw"

get_simd_flags() {
    local profile=$1
    case $profile in
        sse)
            echo "$SIMD_FLAGS_SSE"
            ;;
        avx2)
            echo "$SIMD_FLAGS_AVX2"
            ;;
        avx512)
            echo "$SIMD_FLAGS_AVX512"
            ;;
    esac
}

get_binary_name() {
    local target=$1
    local simd=$2
    local ext=""
    
    if [[ $target == *"windows"* ]]; then
        ext=".exe"
    fi
    
    echo "${PROJECT_NAME}_${simd}${ext}"
}

get_output_dir() {
    local target=$1
    local simd=$2
    
    if [[ $target == *"linux"* ]]; then
        echo "${TARGET_DIR}/linux-x86-64/${simd}"
    else
        echo "${TARGET_DIR}/windows-x86-64/${simd}"
    fi
}

check_target() {
    local target=$1
    if ! rustup target list --installed | grep -q "$target"; then
        echo "Installing target: $target"
        rustup target add "$target"
    fi
}

build_binary() {
    local target=$1
    local simd=$2
    local build_type=$3
    
    local simd_flags=$(get_simd_flags "$simd")
    local output_dir=$(get_output_dir "$target" "$simd")
    local binary_name=$(get_binary_name "$target" "$simd")
    
    echo "Building $PROJECT_NAME for $target with $simd..."
    
    check_target "$target"
    
    mkdir -p "$output_dir"
    
    if [[ $build_type == "release" ]]; then
        RUSTFLAGS="$simd_flags" cargo build --release --target "$target"
        local source_binary="target/${target}/release/${PROJECT_NAME}"
    else
        RUSTFLAGS="$simd_flags" cargo build --target "$target"
        local source_binary="target/${target}/debug/${PROJECT_NAME}"
    fi
    
    if [[ $target == *"windows"* ]]; then
        source_binary="${source_binary}.exe"
    fi
    
    cp "$source_binary" "${output_dir}/${binary_name}"
    
    echo "Built: ${output_dir}/${binary_name}"
}

for build_type in "${BUILD_TYPES[@]}"; do
    for target in "${TARGETS[@]}"; do
        for simd in "${SIMD_PROFILES[@]}"; do
            build_binary "$target" "$simd" "$build_type"
            echo ""
        done
    done
done

echo "=== Build complete ==="
echo "Binaries available in: $TARGET_DIR/"
echo ""
echo "Directory structure:"
find "$TARGET_DIR" -type f -name "${PROJECT_NAME}*" | sort
