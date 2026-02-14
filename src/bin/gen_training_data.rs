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
use std::fs::File;
use std::io::{LineWriter, Write};
use std::time::Instant;

use shallow_guess::chess_position::ChessPosition;
use shallow_guess::def::{A1, BLACK, H8, NO_PIECE, WHITE};
use shallow_guess::generated::network_weights::INPUT_LAYER_SIZE;
use shallow_guess::network::{calculate_network_input_layer_index, FastNoOpNetwork, Network};
use shallow_guess::types::{ChessPiece, ChessSquare};
use shallow_guess::util::{
    read_lines, NetworkInputs, FLIPPED_CHESS_SQUARES, MIRRORED_CHESS_PIECES,
};

const ONE_SYMBOL: &str = "X";

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!(
            "Usage: {} <processed_fen_file> <output_file> <batch_size>",
            args[0]
        );
        std::process::exit(1);
    }

    let processed_fen_file = &args[1];
    let output_file = &args[2];
    let batch_size = args[3].parse::<usize>().unwrap();

    generate_training_set(processed_fen_file, output_file, batch_size);
}

fn generate_training_set(filtered_fen_file: &str, output_file_path: &str, batch_count: usize) {
    let output_file = File::create(output_file_path).unwrap();
    let mut file_writer = LineWriter::new(output_file);

    let mut training_size = 0;
    let mut buffered_line_count = 0;
    let mut output_batch_buffer = String::new();
    let mut chess_position = ChessPosition::new(FastNoOpNetwork::new());

    let start_time = Instant::now();

    if let Ok(lines) = read_lines(filtered_fen_file) {
        for line in lines.flatten() {
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            let splits = line.split(',').collect::<Vec<&str>>();
            let fen = splits[0].trim();
            let mut result: f64 = splits[1].parse().unwrap();

            chess_position.set_from_fen(fen);

            if chess_position.player == BLACK {
                result = 1. - result;
            }

            let network_inputs = parse_network_inputs_from_fen(&chess_position);

            let mut zero_count = 0;

            for x in network_inputs {
                if x == 0 {
                    zero_count += 1;
                } else {
                    if zero_count > 0 {
                        output_batch_buffer.push_str(&format!("{}", zero_count));
                        zero_count = 0;
                    }

                    output_batch_buffer.push_str(ONE_SYMBOL);
                }
            }

            if zero_count > 0 {
                output_batch_buffer.push_str(&format!("{}", zero_count));
            }

            output_batch_buffer.push_str(&format!(",{}\n", result));
            training_size += 1;
            buffered_line_count += 1;

            if buffered_line_count > batch_count {
                println!(
                    "Processed {} positions with PPS {}",
                    training_size,
                    training_size / start_time.elapsed().as_secs().max(1)
                );

                file_writer.write(output_batch_buffer.as_bytes()).unwrap();
                output_batch_buffer.clear();
                buffered_line_count = 0;
            }
        }

        file_writer.write(output_batch_buffer.as_bytes()).unwrap();
    }

    println!(
        "Completed generating training set with size {} to: {}, time taken {}seconds",
        training_size,
        output_file_path,
        start_time.elapsed().as_secs()
    );

    println!("Final training set size: {}", training_size);
}

fn parse_network_inputs_from_fen<N: Network>(chess_position: &ChessPosition<N>) -> NetworkInputs {
    let mut network_inputs = vec![0; INPUT_LAYER_SIZE];

    if chess_position.player == WHITE {
        for current_square in A1..=H8 {
            let piece = chess_position.board[current_square];

            if piece == NO_PIECE {
                continue;
            }

            update_training_network_inputs(&mut network_inputs, piece, current_square);
        }
    } else {
        for current_square in A1..=H8 {
            let piece = chess_position.board[current_square];

            if piece == NO_PIECE {
                continue;
            }

            update_mirrored_training_network_inputs(&mut network_inputs, piece, current_square);
        }
    }

    network_inputs
}

fn update_training_network_inputs(
    network_inputs: &mut NetworkInputs,
    chess_piece: ChessPiece,
    chess_square: ChessSquare,
) {
    network_inputs[calculate_network_input_layer_index(chess_piece, chess_square)] = 1;
}

fn update_mirrored_training_network_inputs(
    network_inputs: &mut NetworkInputs,
    chess_piece: ChessPiece,
    chess_square: ChessSquare,
) {
    network_inputs[calculate_network_input_layer_index(
        MIRRORED_CHESS_PIECES[chess_piece as usize],
        FLIPPED_CHESS_SQUARES[chess_square],
    )] = 1;
}
