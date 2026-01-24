// Copyright 2026 Zixiao Han
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::def::{CHESS_SQUARE_COUNT, WHITE};
use crate::generated::network_weights::{
    HIDDEN_LAYER_BIASES, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_TO_OUTPUT_LAYER_WEIGHTS, INPUT_LAYER_SIZE,
    INPUT_LAYER_TO_HIDDEN_LAYER_WEIGHTS, OUTPUT_BIAS, SCALING_FACTOR,
};
use crate::types::{ChessPiece, ChessSquare, Player, Score};
use crate::util::{
    win_probability_to_centi_pawn_score, FLIPPED_CHESS_SQUARES, MIRRORED_CHESS_PIECES,
};
use std::simd::prelude::*;

pub type NetworkIntValue = i16;
pub type NetworkFloatValue = f32;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const SIMD_F32_LANE_WIDTH: usize = 16;

#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx"),
    not(target_feature = "avx512f")
))]
const SIMD_F32_LANE_WIDTH: usize = 8;

#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "sse4.1", target_feature = "sse2"),
    not(any(
        target_feature = "avx2",
        target_feature = "avx",
        target_feature = "avx512f"
    ))
))]
const SIMD_F32_LANE_WIDTH: usize = 4;

#[cfg(target_arch = "aarch64")]
const SIMD_F32_LANE_WIDTH: usize = 4;

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
const SIMD_F32_LANE_WIDTH: usize = 4;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
const SIMD_I16_LANE_WIDTH: usize = 32;

#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx"),
    not(target_feature = "avx512f")
))]
const SIMD_I16_LANE_WIDTH: usize = 16;

#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "sse4.1", target_feature = "sse2"),
    not(any(
        target_feature = "avx2",
        target_feature = "avx",
        target_feature = "avx512f"
    ))
))]
const SIMD_I16_LANE_WIDTH: usize = 8;

#[cfg(target_arch = "aarch64")]
const SIMD_I16_LANE_WIDTH: usize = 8;

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
const SIMD_I16_LANE_WIDTH: usize = 8;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
type SimdF32 = f32x16;

#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx"),
    not(target_feature = "avx512f")
))]
type SimdF32 = f32x8;

#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "sse4.1", target_feature = "sse2"),
    not(any(
        target_feature = "avx2",
        target_feature = "avx",
        target_feature = "avx512f"
    ))
))]
type SimdF32 = f32x4;

#[cfg(target_arch = "aarch64")]
type SimdF32 = f32x4;

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
type SimdF32 = f32x4;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
type SimdI16 = i16x32;

#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "avx2", target_feature = "avx"),
    not(target_feature = "avx512f")
))]
type SimdI16 = i16x16;

#[cfg(all(
    target_arch = "x86_64",
    any(target_feature = "sse4.1", target_feature = "sse2"),
    not(any(
        target_feature = "avx2",
        target_feature = "avx",
        target_feature = "avx512f"
    ))
))]
type SimdI16 = i16x8;

#[cfg(target_arch = "aarch64")]
type SimdI16 = i16x8;

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
type SimdI16 = i16x8;

#[inline(always)]
pub const fn calculate_network_input_layer_index(
    chess_piece: ChessPiece,
    chess_square: ChessSquare,
) -> usize {
    (chess_piece as usize - 1) * CHESS_SQUARE_COUNT + chess_square
}

#[inline(always)]
fn relu(x: NetworkFloatValue) -> NetworkFloatValue {
    x.max(0.)
}

#[inline(always)]
fn sigmoid(x: NetworkFloatValue) -> NetworkFloatValue {
    1.0 / (1.0 + (-x).exp())
}

pub trait Network: Send + Sync {
    fn add(&mut self, chess_piece: ChessPiece, chess_square: ChessSquare);
    fn remove(&mut self, chess_piece: ChessPiece, chess_square: ChessSquare);
    fn clear_accumulated_layer(&mut self);
    fn evaluate(&self, player: Player) -> Score;
}

pub struct FastNoOpNetwork {}

impl FastNoOpNetwork {
    pub fn new() -> Self {
        Self {}
    }
}

impl Network for FastNoOpNetwork {
    fn add(&mut self, _chess_piece: ChessPiece, _chess_square: ChessSquare) {}

    fn remove(&mut self, _chess_piece: ChessPiece, _chess_square: ChessSquare) {}

    fn clear_accumulated_layer(&mut self) {}

    fn evaluate(&self, _player: Player) -> Score {
        0
    }
}

pub struct QuantizedNetwork {
    transposed_input_layer_to_hidden_layer_weights:
        [[NetworkIntValue; HIDDEN_LAYER_SIZE]; INPUT_LAYER_SIZE],
    hidden_layer_biases: [NetworkFloatValue; HIDDEN_LAYER_SIZE],
    hidden_layer_to_output_layer_weights: [NetworkFloatValue; HIDDEN_LAYER_SIZE],
    output_layer_biases: NetworkFloatValue,

    scaling_factor: NetworkFloatValue,

    white_accumulated_layer: [NetworkIntValue; HIDDEN_LAYER_SIZE],
    black_accumulated_layer: [NetworkIntValue; HIDDEN_LAYER_SIZE],
}

impl QuantizedNetwork {
    pub fn new() -> Self {
        let mut new_network = Self {
            transposed_input_layer_to_hidden_layer_weights: [[0; HIDDEN_LAYER_SIZE];
                INPUT_LAYER_SIZE],
            hidden_layer_biases: [0.; HIDDEN_LAYER_SIZE],
            hidden_layer_to_output_layer_weights: [0.; HIDDEN_LAYER_SIZE],
            output_layer_biases: 0.,

            scaling_factor: 0.,

            white_accumulated_layer: [0; HIDDEN_LAYER_SIZE],
            black_accumulated_layer: [0; HIDDEN_LAYER_SIZE],
        };

        load_default_weights_and_biases(&mut new_network);
        new_network
    }

    #[inline(always)]
    fn evaluate_hidden_layer_with_dot_product(
        &self,
        accumulated: &[NetworkIntValue; HIDDEN_LAYER_SIZE],
    ) -> NetworkFloatValue {
        let scaling = SimdF32::splat(self.scaling_factor);
        let zero = SimdF32::splat(0.0);

        let chunks = HIDDEN_LAYER_SIZE / SIMD_F32_LANE_WIDTH;

        let mut dot_product_vec = SimdF32::splat(0.0);

        for chunk in 0..chunks {
            let base = chunk * SIMD_F32_LANE_WIDTH;

            let mut acc_f32_arr = [0.0f32; SIMD_F32_LANE_WIDTH];
            for i in 0..SIMD_F32_LANE_WIDTH {
                acc_f32_arr[i] = accumulated[base + i] as f32;
            }
            let acc_f32 = SimdF32::from_slice(&acc_f32_arr);

            let bias =
                SimdF32::from_slice(&self.hidden_layer_biases[base..base + SIMD_F32_LANE_WIDTH]);
            let weights = SimdF32::from_slice(
                &self.hidden_layer_to_output_layer_weights[base..base + SIMD_F32_LANE_WIDTH],
            );

            let hidden = (acc_f32 * scaling + bias).simd_max(zero);
            dot_product_vec += hidden * weights;
        }

        let mut dot_product = dot_product_vec.reduce_sum();

        for i in (chunks * SIMD_F32_LANE_WIDTH)..HIDDEN_LAYER_SIZE {
            let hidden = relu(
                accumulated[i] as NetworkFloatValue * self.scaling_factor
                    + self.hidden_layer_biases[i],
            );
            dot_product += hidden * self.hidden_layer_to_output_layer_weights[i];
        }

        dot_product
    }

    fn load_un_flatten(
        &mut self,
        flattened_input_layer_to_hidden_layer_weights: Vec<NetworkIntValue>,
        hidden_layer_biases: Vec<NetworkFloatValue>,
        flattened_hidden_layer_to_output_layer_weights: Vec<NetworkFloatValue>,
        output_biases: Vec<NetworkFloatValue>,
        scaling_factor: NetworkFloatValue,
    ) {
        let mut offset = 0;

        let mut input_layer_to_hidden_layer_weights = [[0; INPUT_LAYER_SIZE]; HIDDEN_LAYER_SIZE];

        for i in 0..HIDDEN_LAYER_SIZE {
            for j in 0..INPUT_LAYER_SIZE {
                input_layer_to_hidden_layer_weights[i][j] =
                    flattened_input_layer_to_hidden_layer_weights[offset];
                offset += 1;
            }
        }

        for i in 0..INPUT_LAYER_SIZE {
            for j in 0..HIDDEN_LAYER_SIZE {
                self.transposed_input_layer_to_hidden_layer_weights[i][j] =
                    input_layer_to_hidden_layer_weights[j][i];
            }
        }

        self.hidden_layer_biases
            .copy_from_slice(&hidden_layer_biases);

        self.hidden_layer_to_output_layer_weights
            .copy_from_slice(&flattened_hidden_layer_to_output_layer_weights);

        self.output_layer_biases = output_biases[0];

        self.scaling_factor = scaling_factor;
    }
}

impl Network for QuantizedNetwork {
    fn add(&mut self, chess_piece: ChessPiece, chess_square: ChessSquare) {
        let index_white_perspective =
            calculate_network_input_layer_index(chess_piece, chess_square);

        let chunks = HIDDEN_LAYER_SIZE / SIMD_I16_LANE_WIDTH;

        for chunk in 0..chunks {
            let base = chunk * SIMD_I16_LANE_WIDTH;
            let acc = SimdI16::from_slice(
                &self.white_accumulated_layer[base..base + SIMD_I16_LANE_WIDTH],
            );
            let weights = SimdI16::from_slice(
                &self.transposed_input_layer_to_hidden_layer_weights[index_white_perspective]
                    [base..base + SIMD_I16_LANE_WIDTH],
            );
            (acc + weights)
                .copy_to_slice(&mut self.white_accumulated_layer[base..base + SIMD_I16_LANE_WIDTH]);
        }

        for i in (chunks * SIMD_I16_LANE_WIDTH)..HIDDEN_LAYER_SIZE {
            self.white_accumulated_layer[i] +=
                self.transposed_input_layer_to_hidden_layer_weights[index_white_perspective][i];
        }

        let index_black_perspective = calculate_network_input_layer_index(
            MIRRORED_CHESS_PIECES[chess_piece as usize],
            FLIPPED_CHESS_SQUARES[chess_square],
        );

        for chunk in 0..chunks {
            let base = chunk * SIMD_I16_LANE_WIDTH;
            let acc = SimdI16::from_slice(
                &self.black_accumulated_layer[base..base + SIMD_I16_LANE_WIDTH],
            );
            let weights = SimdI16::from_slice(
                &self.transposed_input_layer_to_hidden_layer_weights[index_black_perspective]
                    [base..base + SIMD_I16_LANE_WIDTH],
            );
            (acc + weights)
                .copy_to_slice(&mut self.black_accumulated_layer[base..base + SIMD_I16_LANE_WIDTH]);
        }

        for i in (chunks * SIMD_I16_LANE_WIDTH)..HIDDEN_LAYER_SIZE {
            self.black_accumulated_layer[i] +=
                self.transposed_input_layer_to_hidden_layer_weights[index_black_perspective][i];
        }
    }

    fn remove(&mut self, chess_piece: ChessPiece, chess_square: ChessSquare) {
        let index_white_perspective =
            calculate_network_input_layer_index(chess_piece, chess_square);

        let chunks = HIDDEN_LAYER_SIZE / SIMD_I16_LANE_WIDTH;

        for chunk in 0..chunks {
            let base = chunk * SIMD_I16_LANE_WIDTH;
            let acc = SimdI16::from_slice(
                &self.white_accumulated_layer[base..base + SIMD_I16_LANE_WIDTH],
            );
            let weights = SimdI16::from_slice(
                &self.transposed_input_layer_to_hidden_layer_weights[index_white_perspective]
                    [base..base + SIMD_I16_LANE_WIDTH],
            );
            (acc - weights)
                .copy_to_slice(&mut self.white_accumulated_layer[base..base + SIMD_I16_LANE_WIDTH]);
        }

        for i in (chunks * SIMD_I16_LANE_WIDTH)..HIDDEN_LAYER_SIZE {
            self.white_accumulated_layer[i] -=
                self.transposed_input_layer_to_hidden_layer_weights[index_white_perspective][i];
        }

        let index_black_perspective = calculate_network_input_layer_index(
            MIRRORED_CHESS_PIECES[chess_piece as usize],
            FLIPPED_CHESS_SQUARES[chess_square],
        );

        for chunk in 0..chunks {
            let base = chunk * SIMD_I16_LANE_WIDTH;
            let acc = SimdI16::from_slice(
                &self.black_accumulated_layer[base..base + SIMD_I16_LANE_WIDTH],
            );
            let weights = SimdI16::from_slice(
                &self.transposed_input_layer_to_hidden_layer_weights[index_black_perspective]
                    [base..base + SIMD_I16_LANE_WIDTH],
            );
            (acc - weights)
                .copy_to_slice(&mut self.black_accumulated_layer[base..base + SIMD_I16_LANE_WIDTH]);
        }

        for i in (chunks * SIMD_I16_LANE_WIDTH)..HIDDEN_LAYER_SIZE {
            self.black_accumulated_layer[i] -=
                self.transposed_input_layer_to_hidden_layer_weights[index_black_perspective][i];
        }
    }

    fn clear_accumulated_layer(&mut self) {
        self.white_accumulated_layer = [0; HIDDEN_LAYER_SIZE];
        self.black_accumulated_layer = [0; HIDDEN_LAYER_SIZE];
    }

    fn evaluate(&self, player: Player) -> Score {
        let accumulated_layer = if player == WHITE {
            &self.white_accumulated_layer
        } else {
            &self.black_accumulated_layer
        };

        let dot_product = self.evaluate_hidden_layer_with_dot_product(accumulated_layer);
        let output = self.output_layer_biases + dot_product;

        let win_probability = sigmoid(output);
        win_probability_to_centi_pawn_score(win_probability)
    }
}

fn load_default_weights_and_biases(network: &mut QuantizedNetwork) {
    let input_weights_i16: Vec<NetworkIntValue> = INPUT_LAYER_TO_HIDDEN_LAYER_WEIGHTS
        .iter()
        .map(|&x| x as NetworkIntValue)
        .collect();

    let flattened_hidden_to_output = HIDDEN_LAYER_TO_OUTPUT_LAYER_WEIGHTS.to_vec();

    network.load_un_flatten(
        input_weights_i16,
        HIDDEN_LAYER_BIASES.to_vec(),
        flattened_hidden_to_output,
        vec![OUTPUT_BIAS],
        SCALING_FACTOR,
    );
}
