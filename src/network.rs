// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

use crate::def::{CHESS_SQUARE_COUNT, TERMINATE_SCORE, WHITE};
use crate::network_weights::{
    HIDDEN_LAYER_BIASES, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_TO_OUTPUT_LAYER_WEIGHTS, INPUT_LAYER_SIZE,
    INPUT_LAYER_TO_HIDDEN_LAYER_WEIGHTS, OUTPUT_BIASES, OUTPUT_LAYER_SIZE, SCALING_FACTOR,
};
use crate::types::{ChessPiece, ChessSquare, Player, Score};
use crate::util::{FLIPPED_CHESS_SQUARES, MIRRORED_CHESS_PIECES};
use std::simd::prelude::*;

pub type NetworkIntValue = i16;
pub type NetworkFloatValue = f32;

const CENTI_PAWN_SCORE_SCALING_FACTOR: f32 = 0.004;
const WIN_PROBABILITY_EPSILON: f32 = 1e-7;

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
fn softmax(
    logits: &[NetworkFloatValue; OUTPUT_LAYER_SIZE],
) -> [NetworkFloatValue; OUTPUT_LAYER_SIZE] {
    let mut max_logit = logits[0];
    for i in 1..OUTPUT_LAYER_SIZE {
        if logits[i] > max_logit {
            max_logit = logits[i];
        }
    }

    let mut sum_exp = 0.0;
    let mut exps = [0.0; OUTPUT_LAYER_SIZE];

    for i in 0..OUTPUT_LAYER_SIZE {
        exps[i] = (logits[i] - max_logit).exp();
        sum_exp += exps[i];
    }

    let mut probs = [0.0; OUTPUT_LAYER_SIZE];
    for i in 0..OUTPUT_LAYER_SIZE {
        probs[i] = exps[i] / sum_exp;
    }

    probs
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
    hidden_layer_to_output_layer_weights:
        [[NetworkFloatValue; HIDDEN_LAYER_SIZE]; OUTPUT_LAYER_SIZE],
    output_layer_biases: [NetworkFloatValue; OUTPUT_LAYER_SIZE],

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
            hidden_layer_to_output_layer_weights: [[0.; HIDDEN_LAYER_SIZE]; OUTPUT_LAYER_SIZE],
            output_layer_biases: [0.; OUTPUT_LAYER_SIZE],

            scaling_factor: 0.,

            white_accumulated_layer: [0; HIDDEN_LAYER_SIZE],
            black_accumulated_layer: [0; HIDDEN_LAYER_SIZE],
        };

        load_default_weights_and_biases(&mut new_network);
        new_network
    }

    #[inline(always)]
    fn evaluate_with_logits(&self, player: Player) -> NetworkFloatValue {
        let accumulated_layer = if player == WHITE {
            &self.white_accumulated_layer
        } else {
            &self.black_accumulated_layer
        };

        let mut hidden_layer = [0.; HIDDEN_LAYER_SIZE];

        self.evaluate_hidden_layer(accumulated_layer, &mut hidden_layer);

        let mut logits = [0.0; OUTPUT_LAYER_SIZE];

        for k in 0..OUTPUT_LAYER_SIZE {
            let mut output = self.output_layer_biases[k];
            let weights_row = &self.hidden_layer_to_output_layer_weights[k];

            output += self.dot_product(&hidden_layer, weights_row);
            logits[k] = output;
        }

        let probs = softmax(&logits);
        probs[2] + probs[1] * 0.5
    }

    #[inline(always)]
    fn evaluate_hidden_layer(
        &self,
        accumulated: &[NetworkIntValue; HIDDEN_LAYER_SIZE],
        hidden: &mut [NetworkFloatValue; HIDDEN_LAYER_SIZE],
    ) {
        let scaling = SimdF32::splat(self.scaling_factor);
        let zero = SimdF32::splat(0.0);

        let chunks = HIDDEN_LAYER_SIZE / SIMD_F32_LANE_WIDTH;

        for chunk in 0..chunks {
            let base = chunk * SIMD_F32_LANE_WIDTH;

            // Convert i16 accumulated values to f32 for SIMD processing
            let mut acc_f32_arr = [0.0f32; SIMD_F32_LANE_WIDTH];
            for i in 0..SIMD_F32_LANE_WIDTH {
                acc_f32_arr[i] = accumulated[base + i] as f32;
            }
            let acc_f32 = SimdF32::from_slice(&acc_f32_arr);

            let bias =
                SimdF32::from_slice(&self.hidden_layer_biases[base..base + SIMD_F32_LANE_WIDTH]);

            let result = (acc_f32 * scaling + bias).simd_max(zero);

            result.copy_to_slice(&mut hidden[base..base + SIMD_F32_LANE_WIDTH]);
        }

        for i in (chunks * SIMD_F32_LANE_WIDTH)..HIDDEN_LAYER_SIZE {
            hidden[i] = relu(
                accumulated[i] as NetworkFloatValue * self.scaling_factor
                    + self.hidden_layer_biases[i],
            );
        }
    }

    #[inline(always)]
    fn dot_product(&self, a: &[NetworkFloatValue], b: &[NetworkFloatValue]) -> NetworkFloatValue {
        let len = a.len();
        let chunks = len / SIMD_F32_LANE_WIDTH;

        let mut sum_vec = SimdF32::splat(0.0);

        for chunk in 0..chunks {
            let base = chunk * SIMD_F32_LANE_WIDTH;
            let a_vec = SimdF32::from_slice(&a[base..base + SIMD_F32_LANE_WIDTH]);
            let b_vec = SimdF32::from_slice(&b[base..base + SIMD_F32_LANE_WIDTH]);
            sum_vec += a_vec * b_vec;
        }

        let mut sum = sum_vec.reduce_sum();

        for i in (chunks * SIMD_F32_LANE_WIDTH)..len {
            sum += a[i] * b[i];
        }

        sum
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

        for k in 0..OUTPUT_LAYER_SIZE {
            for i in 0..HIDDEN_LAYER_SIZE {
                let idx = k * HIDDEN_LAYER_SIZE + i;
                self.hidden_layer_to_output_layer_weights[k][i] =
                    flattened_hidden_layer_to_output_layer_weights[idx];
            }
        }

        self.output_layer_biases.copy_from_slice(&output_biases);

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
        win_probability_to_centi_pawn_score(self.evaluate_with_logits(player))
    }
}

fn win_probability_to_centi_pawn_score(win_probability: NetworkFloatValue) -> Score {
    let clamped = win_probability.clamp(WIN_PROBABILITY_EPSILON, 1.0 - WIN_PROBABILITY_EPSILON);
    let odds = clamped / (1.0 - clamped);
    let log_odds = odds.ln();

    let score = (log_odds / CENTI_PAWN_SCORE_SCALING_FACTOR) as Score;

    score.min(TERMINATE_SCORE).max(-TERMINATE_SCORE)
}

fn load_default_weights_and_biases(network: &mut QuantizedNetwork) {
    let input_weights_i16: Vec<NetworkIntValue> = INPUT_LAYER_TO_HIDDEN_LAYER_WEIGHTS
        .iter()
        .map(|&x| x as NetworkIntValue)
        .collect();

    let mut flattened_hidden_to_output = Vec::with_capacity(HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE);
    for i in 0..OUTPUT_LAYER_SIZE {
        for j in 0..HIDDEN_LAYER_SIZE {
            flattened_hidden_to_output.push(HIDDEN_LAYER_TO_OUTPUT_LAYER_WEIGHTS[i][j]);
        }
    }

    network.load_un_flatten(
        input_weights_i16,
        HIDDEN_LAYER_BIASES.to_vec(),
        flattened_hidden_to_output,
        OUTPUT_BIASES.to_vec(),
        SCALING_FACTOR,
    );
}
