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
use std::option::Option;
use std::time::Instant;

use shallow_guess::chess_move_gen::{
    generate_captures_and_promotions, is_in_check, is_invalid_position,
};
use shallow_guess::chess_position::ChessPosition;
use shallow_guess::def::{A1, BLACK, H8, NO_PIECE, PIECE_TYPE_COUNT, WHITE};
use shallow_guess::generated::network_weights::INPUT_LAYER_SIZE;
use shallow_guess::network::{calculate_network_input_layer_index, FastNoOpNetwork, Network};
use shallow_guess::types::{ChessPiece, ChessSquare, Score, SearchPly};
use shallow_guess::util::{
    read_lines, NetworkInputs, FLIPPED_CHESS_SQUARES, MIRRORED_CHESS_PIECES,
};

const ONE_SYMBOL: &str = "X";

const MATERIAL_SCORES: [Score; PIECE_TYPE_COUNT] =
    [0, 1, 3, 3, 5, 10, 100, -1, -3, -3, -5, -10, -100];

const MAX_EXCHANGE_SEARCH_PLY: SearchPly = 8;

fn main() {
    let mut args = env::args().into_iter();
    args.next().unwrap();

    let processed_fen_file = args.next().unwrap();
    let output_file = args.next().unwrap();
    let batch_size = args.next().unwrap().parse::<usize>().unwrap();
    let log_file = args.next();

    generate_training_set(&processed_fen_file, &output_file, batch_size, log_file);
}

fn generate_training_set(
    processed_fen_file: &str,
    output_file_path: &str,
    batch_count: usize,
    log_file_path: Option<String>,
) {
    let output_file = File::create(output_file_path).unwrap();
    let mut file_writer = LineWriter::new(output_file);

    let mut log_writer: Option<LineWriter<File>> = None;
    if let Some(log_path) = log_file_path {
        let log_file = File::create(log_path).unwrap();
        log_writer = Some(LineWriter::new(log_file));
    }

    let mut training_size = 0;
    let mut buffered_line_count = 0;
    let mut output_batch_buffer = String::new();

    let mut total_positions = 0;
    let mut non_static_filtered_count = 0;
    let mut static_kept_count = 0;

    let start_time = Instant::now();

    if let Ok(lines) = read_lines(processed_fen_file) {
        for line in lines.flatten() {
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            let splits = line.split(',').collect::<Vec<&str>>();
            let fen = splits[0];

            let mut chess_position = ChessPosition::new(FastNoOpNetwork::new());
            chess_position.set_from_fen(fen);

            total_positions += 1;

            if is_in_check(&chess_position) {
                if let Some(ref mut writer) = log_writer {
                    writeln!(writer, "Filtered In-check position {}", fen).unwrap();
                }
                non_static_filtered_count += 1;
                continue;
            }

            let static_score = get_material_score(&chess_position);
            let q_score = exchange_search(&mut chess_position, static_score, static_score + 1, 0);

            if q_score != static_score {
                if let Some(ref mut writer) = log_writer {
                    writeln!(
                        writer,
                        "Filtered Non-static position {}, static_score: {}, q_score: {}",
                        fen, static_score, q_score
                    )
                    .unwrap();
                }
                non_static_filtered_count += 1;
                continue;
            } else {
                if let Some(ref mut writer) = log_writer {
                    writeln!(
                        writer,
                        "Keeping Static position {}, static_score: {}, q_score: {}",
                        fen, static_score, q_score
                    )
                    .unwrap();
                }
                static_kept_count += 1;
            }

            let network_inputs = parse_network_inputs_from_fen(&chess_position);
            let mut result = splits[1].parse::<f64>().unwrap();

            if chess_position.player == BLACK {
                result = 1. - result;
            }

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

    println!("Total positions processed: {}", total_positions);
    println!("Positions filtered: {}", non_static_filtered_count);
    println!("Positions kept: {}", static_kept_count);
    println!("Final training set size: {}", training_size);
}

fn exchange_search<N: Network>(
    chess_position: &mut ChessPosition<N>,
    mut alpha: Score,
    beta: Score,
    ply: SearchPly,
) -> Score {
    let static_eval = get_material_score(chess_position);

    if ply > MAX_EXCHANGE_SEARCH_PLY {
        return static_eval;
    }

    if static_eval >= beta {
        return static_eval;
    }

    if static_eval > alpha {
        alpha = static_eval;
    }

    let mut captures_and_promotions = generate_captures_and_promotions(chess_position);

    while let Some(chess_move) = captures_and_promotions.pop() {
        let saved_state = chess_position.make_move(&chess_move);

        if is_invalid_position(chess_position) {
            chess_position.unmake_move(&chess_move, saved_state);
            continue;
        }

        let score = -exchange_search(chess_position, -beta, -alpha, ply + 1);

        chess_position.unmake_move(&chess_move, saved_state);

        if score >= beta {
            return score;
        }

        if score > alpha {
            alpha = score;
        }
    }

    alpha
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

fn get_material_score<N: Network>(chess_position: &ChessPosition<N>) -> Score {
    let mut score = 0;

    for chess_square in A1..=H8 {
        score += MATERIAL_SCORES[chess_position.board[chess_square] as usize];
    }

    if chess_position.player == WHITE {
        score
    } else {
        -score
    }
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
