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

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

use shallow_guess::def::{
    get_bucket_index, A1, BB, BK, BN, BP, BQ, BR, BUCKET_COUNT, CHESS_RANK_COUNT,
    CHESS_SQUARE_COUNT, H8, PIECE_TYPE_COUNT, QUEEN_PHASE, ROOK_PHASE, WK, WP, WQ,
};
use shallow_guess::generated::network_weights::INPUT_LAYER_SIZE;
use shallow_guess::util::{read_lines, FLIPPED_CHESS_SQUARES, MIRRORED_CHESS_PIECES};

const OPENING_PAWN_CENTIPAWN_VAL: f64 = 100.;
const HUBER_DELTA: f64 = 0.3;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn huber_loss(error: f64) -> f64 {
    let abs_error = error.abs();
    if abs_error <= HUBER_DELTA {
        0.5 * error * error
    } else {
        HUBER_DELTA * (abs_error - 0.5 * HUBER_DELTA)
    }
}

fn huber_gradient(error: f64) -> f64 {
    let abs_error = error.abs();
    if abs_error <= HUBER_DELTA {
        error
    } else {
        HUBER_DELTA * error.signum()
    }
}

fn decode_network_inputs(encoded: &str) -> Vec<f64> {
    let mut inputs = vec![0.0; INPUT_LAYER_SIZE];
    let mut pos = 0;
    let mut current_number = String::new();

    for ch in encoded.chars() {
        if ch == 'X' {
            if !current_number.is_empty() {
                pos += current_number.parse::<usize>().unwrap();
                current_number.clear();
            }

            if pos < INPUT_LAYER_SIZE {
                inputs[pos] = 1.0;
            }

            pos += 1;
        } else if ch.is_ascii_digit() {
            current_number.push(ch);
        }
    }

    inputs
}

fn count_pieces(inputs: &[f64]) -> [u8; PIECE_TYPE_COUNT] {
    let mut counts = [0u8; PIECE_TYPE_COUNT];

    for chess_piece in WP..=WQ {
        let piece_index = chess_piece as usize;

        let start = (piece_index - 1) * CHESS_SQUARE_COUNT;
        let end = start + CHESS_SQUARE_COUNT;

        for index in start..end {
            if inputs[index] != 0.0 {
                counts[piece_index] += 1;
            }
        }
    }

    for chess_piece in BP..=BQ {
        let piece_index = chess_piece as usize;

        let start = (piece_index - 1) * CHESS_SQUARE_COUNT;
        let end = start + CHESS_SQUARE_COUNT;

        for index in start..end {
            if inputs[index] != 0.0 {
                counts[piece_index] += 1;
            }
        }
    }

    counts
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 6 || args.len() > 7 {
        eprintln!(
            "Usage: {} <input_dir> <learning_rate> <batch_size> <save_cycle> <output_weights> [input_weights]",
            args[0]
        );
        std::process::exit(1);
    }

    let input_dir = &args[1];
    let learning_rate: f64 = args[2].parse().expect("Invalid learning rate");
    let batch_size: usize = args[3].parse().expect("Invalid batch size");
    let save_cycle: usize = args[4].parse().expect("Invalid save cycle");
    let output_weights_path = &args[5];
    let input_weights_path = args.get(6).map(|s| s.as_str());
    let output_path = "src/generated/piece_square_table.rs";
    let piece_values_path = "src/generated/piece_values.rs";

    let mut piece_weights: [[f64; PIECE_TYPE_COUNT]; BUCKET_COUNT] =
        [[0.; PIECE_TYPE_COUNT]; BUCKET_COUNT];
    let mut square_weights: [[f64; INPUT_LAYER_SIZE]; BUCKET_COUNT] =
        [[0.; INPUT_LAYER_SIZE]; BUCKET_COUNT];

    let mut piece_grad_accum: [[f64; PIECE_TYPE_COUNT]; BUCKET_COUNT] =
        [[0.; PIECE_TYPE_COUNT]; BUCKET_COUNT];
    let mut square_grad_accum: [[f64; INPUT_LAYER_SIZE]; BUCKET_COUNT] =
        [[0.; INPUT_LAYER_SIZE]; BUCKET_COUNT];

    if let Some(input_path) = input_weights_path {
        if std::path::Path::new(input_path).exists() {
            load_weights(&mut square_weights, &mut piece_weights, input_path);
            println!("Loaded weights from {}", input_path);
        } else {
            eprintln!("Input weights file not found: {}", input_path);
            std::process::exit(1);
        }
    }

    let start_time = Instant::now();
    let mut row_count = 0;
    let mut sample_count = 0;
    let mut batch_count = 0;
    let mut batch_loss = 0.0;
    let mut total_loss = 0.0;

    let files: Vec<_> = fs::read_dir(input_dir)
        .expect("Failed to read input directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .collect();

    if files.is_empty() {
        println!("No files found in input directory");
        return;
    }

    for entry in files {
        let input_file = entry.path();

        if let Ok(lines) = read_lines(input_file.to_str().unwrap()) {
            for line in lines.flatten() {
                let line = line.trim();

                if line.is_empty() {
                    continue;
                }

                let splits: Vec<&str> = line.split(',').collect();
                let encoded = splits[0].trim();
                let weighted_result: f64 = splits[1].parse().unwrap();

                let inputs = decode_network_inputs(encoded);
                let piece_counts = count_pieces(&inputs);

                let game_phase = piece_counts[BN as usize]
                    + piece_counts[BB as usize]
                    + piece_counts[BR as usize] * ROOK_PHASE
                    + piece_counts[BQ as usize] * QUEEN_PHASE;
                let bucket_idx = get_bucket_index(game_phase);

                let square_linear_combination: f64 = square_weights[bucket_idx]
                    .iter()
                    .zip(inputs.iter())
                    .map(|(w, x)| w * x)
                    .sum();

                let piece_linear_combination: f64 = piece_weights[bucket_idx]
                    .iter()
                    .zip(piece_counts.iter())
                    .map(|(w, x)| w * *x as f64)
                    .sum();

                let prediction = sigmoid(piece_linear_combination + square_linear_combination);
                let error = prediction - weighted_result;
                let huber_grad = huber_gradient(error);

                for index in 0..PIECE_TYPE_COUNT {
                    piece_grad_accum[bucket_idx][index] += huber_grad * piece_counts[index] as f64;
                }

                for index in 0..INPUT_LAYER_SIZE {
                    square_grad_accum[bucket_idx][index] += huber_grad * inputs[index];
                }

                row_count += 1;
                sample_count += 1;
                batch_loss += huber_loss(error);
                total_loss += huber_loss(error);

                if sample_count >= batch_size {
                    batch_count += 1;
                    for bucket in 0..BUCKET_COUNT {
                        for index in 0..PIECE_TYPE_COUNT {
                            piece_weights[bucket][index] -=
                                learning_rate * piece_grad_accum[bucket][index] / batch_size as f64;
                            piece_grad_accum[bucket][index] = 0.0;
                        }

                        for index in 0..INPUT_LAYER_SIZE {
                            square_weights[bucket][index] -= learning_rate
                                * square_grad_accum[bucket][index]
                                / batch_size as f64;
                            square_grad_accum[bucket][index] = 0.0;
                        }
                    }

                    if batch_count % save_cycle == 0 {
                        let avg_loss = batch_loss / batch_size as f64;
                        let overall_avg_loss = total_loss / row_count as f64;

                        println!(
                            "Processed {} rows in {:?}, PPS: {:.0}, Batch Loss: {:.6}, Overall Loss: {:.6}",
                            row_count,
                            start_time.elapsed(),
                            row_count as f64 / start_time.elapsed().as_secs_f64(),
                            avg_loss,
                            overall_avg_loss
                        );

                        save_weights(&square_weights, &piece_weights, output_weights_path);
                        output_square_weights(&square_weights, &piece_weights, output_path);
                        output_piece_values(&piece_weights, piece_values_path);

                        println!(
                            "Saved weights to {} at batch {}",
                            output_weights_path, batch_count
                        );
                    }

                    sample_count = 0;
                    batch_loss = 0.0;
                }
            }
        }
    }

    if sample_count > 0 {
        for bucket in 0..BUCKET_COUNT {
            for index in 0..PIECE_TYPE_COUNT {
                piece_weights[bucket][index] -=
                    learning_rate * piece_grad_accum[bucket][index] / sample_count as f64;
            }

            for index in 0..INPUT_LAYER_SIZE {
                square_weights[bucket][index] -=
                    learning_rate * square_grad_accum[bucket][index] / sample_count as f64;
            }
        }
    }

    output_square_weights(&square_weights, &piece_weights, output_path);
    output_piece_values(&piece_weights, piece_values_path);
    save_weights(&square_weights, &piece_weights, output_weights_path);

    println!("Processed {} rows in {:?}", row_count, start_time.elapsed());
    println!("Generated table written to {}", output_path);
    println!("Piece values written to {}", piece_values_path);
    println!("Weights saved to {}", output_weights_path);
}

fn load_weights(
    square_weights: &mut [[f64; INPUT_LAYER_SIZE]; BUCKET_COUNT],
    piece_weights: &mut [[f64; PIECE_TYPE_COUNT]; BUCKET_COUNT],
    path: &str,
) {
    let content = fs::read_to_string(path).expect("Failed to read weights file");
    let lines: Vec<&str> = content.lines().collect();

    if lines.len() != BUCKET_COUNT * 2 {
        panic!("Expected {} lines, got {}", BUCKET_COUNT * 2, lines.len());
    }

    for (bucket_idx, line) in lines.iter().take(BUCKET_COUNT).enumerate() {
        let values: Vec<&str> = line.split(',').collect();

        if values.len() != PIECE_TYPE_COUNT {
            panic!(
                "Line {}: expected {} values, got {}",
                bucket_idx + 1,
                PIECE_TYPE_COUNT,
                values.len()
            );
        }

        for (i, value) in values.iter().enumerate() {
            piece_weights[bucket_idx][i] = value.trim().parse().expect("Invalid float value");
        }
    }

    for (bucket_idx, line) in lines.iter().skip(BUCKET_COUNT).enumerate() {
        let values: Vec<&str> = line.split(',').collect();

        if values.len() != INPUT_LAYER_SIZE {
            panic!(
                "Line {}: expected {} values, got {}",
                BUCKET_COUNT + bucket_idx + 1,
                INPUT_LAYER_SIZE,
                values.len()
            );
        }

        for (i, value) in values.iter().enumerate() {
            square_weights[bucket_idx][i] = value.trim().parse().expect("Invalid float value");
        }
    }
}

fn save_weights(
    square_weights: &[[f64; INPUT_LAYER_SIZE]; BUCKET_COUNT],
    piece_weights: &[[f64; PIECE_TYPE_COUNT]; BUCKET_COUNT],
    path: &str,
) {
    let mut output = String::new();

    for bucket_idx in 0..BUCKET_COUNT {
        let line: Vec<String> = piece_weights[bucket_idx]
            .iter()
            .map(|w| w.to_string())
            .collect();
        output.push_str(&line.join(","));
        output.push('\n');
    }

    for bucket_idx in 0..BUCKET_COUNT {
        let line: Vec<String> = square_weights[bucket_idx]
            .iter()
            .map(|w| w.to_string())
            .collect();
        output.push_str(&line.join(","));
        output.push('\n');
    }

    let mut file = File::create(path).expect("Failed to create weights file");
    file.write_all(output.as_bytes())
        .expect("Failed to write weights file");
}

fn output_square_weights(
    weights: &[[f64; INPUT_LAYER_SIZE]; BUCKET_COUNT],
    piece_weights: &[[f64; PIECE_TYPE_COUNT]; BUCKET_COUNT],
    output_path: &str,
) {
    let mut tables: [[[f64; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT]; BUCKET_COUNT] =
        [[[0.; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT]; BUCKET_COUNT];

    let scale = 100.0 / piece_weights[0][WP as usize];

    for bucket_idx in 0..BUCKET_COUNT {
        for chess_piece in WP..=WK {
            let piece_index = chess_piece as usize;

            for chess_square in A1..=H8 {
                let weight_index = (piece_index - 1) * CHESS_SQUARE_COUNT + chess_square;
                tables[bucket_idx][piece_index][chess_square] =
                    weights[bucket_idx][weight_index] * scale;
            }
        }

        for chess_piece in BP..=BK {
            for chess_square in A1..=H8 {
                tables[bucket_idx][chess_piece as usize][chess_square] = tables[bucket_idx]
                    [MIRRORED_CHESS_PIECES[chess_piece as usize] as usize]
                    [FLIPPED_CHESS_SQUARES[chess_square]];
            }
        }
    }

    let mut output = String::new();

    output.push_str("use crate::def::{BUCKET_COUNT, CHESS_SQUARE_COUNT, PIECE_TYPE_COUNT};\n");
    output.push_str("use crate::types::Score;\n\n");

    output.push_str(&format!(
        "pub const PIECE_SQUARE_TABLE: [[[Score; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT]; BUCKET_COUNT] = [\n"
    ));

    for (bucket_idx, table) in tables.iter().enumerate() {
        output.push_str("    [\n");
        for (piece_idx, piece_table) in table.iter().enumerate() {
            let piece_name = get_piece_name(piece_idx);
            output.push_str(&format!(
                "        // {}/Bucket {}\n        [\n            ",
                piece_name, bucket_idx
            ));
            for (sq_idx, &value) in piece_table.iter().enumerate() {
                output.push_str(&format!("{:5.0}", value));

                if sq_idx < CHESS_SQUARE_COUNT - 1 {
                    output.push_str(", ");
                }

                if (sq_idx + 1) % CHESS_RANK_COUNT == 0 && sq_idx < CHESS_SQUARE_COUNT - 1 {
                    output.push_str("\n            ");
                }
            }

            if piece_idx < PIECE_TYPE_COUNT - 1 {
                output.push_str(",\n        ],\n");
            } else {
                output.push_str(",\n        ]\n");
            }
        }
        if bucket_idx < BUCKET_COUNT - 1 {
            output.push_str("    ],\n");
        } else {
            output.push_str("    ]\n");
        }
    }

    output.push_str("];\n");

    let mut file = File::create(output_path).expect("Failed to create output file");
    file.write_all(output.as_bytes())
        .expect("Failed to write to output file");
}

fn output_piece_values(piece_weights: &[[f64; PIECE_TYPE_COUNT]; BUCKET_COUNT], output_path: &str) {
    let scale = OPENING_PAWN_CENTIPAWN_VAL / piece_weights[BUCKET_COUNT - 1][WP as usize];

    let mut output = String::new();

    output.push_str("use crate::def::{BUCKET_COUNT, PIECE_TYPE_COUNT};\n");
    output.push_str("use crate::types::Score;\n\n");

    output.push_str(&format!(
        "pub const PIECE_VALUES: [[Score; PIECE_TYPE_COUNT]; BUCKET_COUNT] = [\n"
    ));

    for (bucket_idx, bucket_weights) in piece_weights.iter().enumerate() {
        output.push_str("    [");
        for (piece_idx, &weight) in bucket_weights.iter().enumerate() {
            let normalized = weight * scale;
            output.push_str(&format!("{:5.0}", normalized));

            if piece_idx < PIECE_TYPE_COUNT - 1 {
                output.push_str(", ");
            }
        }
        if bucket_idx < BUCKET_COUNT - 1 {
            output.push_str("],\n");
        } else {
            output.push_str("]\n");
        }
    }

    output.push_str("];\n");

    let mut file = File::create(output_path).expect("Failed to create output file");
    file.write_all(output.as_bytes())
        .expect("Failed to write to output file");
}

fn get_piece_name(piece_idx: usize) -> &'static str {
    match piece_idx {
        0 => "NoPiece",
        1 => "WP",
        2 => "WN",
        3 => "WB",
        4 => "WR",
        5 => "WQ",
        6 => "WK",
        7 => "BP",
        8 => "BN",
        9 => "BB",
        10 => "BR",
        11 => "BQ",
        12 => "BK",
        _ => "Unknown",
    }
}
