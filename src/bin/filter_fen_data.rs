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

use shallow_guess::chess_move_gen::{
    generate_captures_and_promotions, is_in_check, is_invalid_position,
};
use shallow_guess::chess_position::ChessPosition;
use shallow_guess::def::{A1, H8, PIECE_VALS_PLAYER_PERSPECTIVE};
use shallow_guess::network::{FastNoOpNetwork, Network};
use shallow_guess::types::{Score, SearchPly};
use shallow_guess::util::read_lines;

const MAX_PLY: SearchPly = 8;

const DRAW_RESULT: f64 = 0.5;
const SMOOTHED_LABEL_RANGE: f64 = 0.45;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!(
            "Usage: {} <preprocessed_fen_file> <output_file> <batch_size>",
            args[0]
        );
        std::process::exit(1);
    }

    let preprocessed_fen_file = &args[1];
    let output_file = &args[2];
    let batch_size = args[3].parse::<usize>().unwrap();

    filter_fen(preprocessed_fen_file, output_file, batch_size);
}

fn filter_fen(preprocessed_fen_file: &str, output_file_path: &str, batch_count: usize) {
    let output_file = File::create(output_file_path).unwrap();
    let mut file_writer = LineWriter::new(output_file);

    let mut training_size = 0;
    let mut buffered_line_count = 0;
    let mut output_batch_buffer = String::new();
    let mut total_positions = 0;
    let mut non_static_filtered_count = 0;
    let mut static_kept_count = 0;
    let mut chess_position = ChessPosition::new(FastNoOpNetwork::new());

    let start_time = Instant::now();

    if let Ok(lines) = read_lines(preprocessed_fen_file) {
        for line in lines.flatten() {
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            let splits = line.split(',').collect::<Vec<&str>>();
            let fen = splits[0];
            let result = splits[1].parse().unwrap();
            let position_count: f64 = splits[2].parse().unwrap();
            let total_position_count: f64 = splits[3].parse().unwrap();

            chess_position.set_from_fen(fen);

            total_positions += 1;

            if is_in_check(&chess_position) {
                non_static_filtered_count += 1;
                continue;
            }

            let material_score = get_material_score(&chess_position);
            let exchange_score =
                exchange_search(&mut chess_position, material_score, material_score + 1, 0);

            if material_score != exchange_score {
                non_static_filtered_count += 1;
                continue;
            } else {
                static_kept_count += 1;
            }

            let weighted_result = match result {
                DRAW_RESULT => DRAW_RESULT,
                1.0 => DRAW_RESULT + SMOOTHED_LABEL_RANGE * (position_count / total_position_count),
                0.0 => DRAW_RESULT - SMOOTHED_LABEL_RANGE * (position_count / total_position_count),
                _ => panic!("Invalid game result {}", result),
            };

            output_batch_buffer.push_str(&format!("{},{}\n", fen, weighted_result));

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
        "Completed filtering fen data with size {} to: {}, time taken {}seconds",
        training_size,
        output_file_path,
        start_time.elapsed().as_secs()
    );

    println!("Total positions processed: {}", total_positions);
    println!("Positions filtered: {}", non_static_filtered_count);
    println!("Positions kept: {}", static_kept_count);
}

fn get_material_score<N: Network>(chess_position: &ChessPosition<N>) -> Score {
    let mut score = 0;

    let player_index = chess_position.player as usize;

    for chess_square in A1..=H8 {
        score += PIECE_VALS_PLAYER_PERSPECTIVE[player_index]
            [chess_position.board[chess_square] as usize];
    }

    score
}

fn exchange_search<N: Network>(
    chess_position: &mut ChessPosition<N>,
    mut alpha: Score,
    beta: Score,
    ply: SearchPly,
) -> Score {
    let material_eval = get_material_score(chess_position);

    if ply >= MAX_PLY {
        return material_eval;
    }

    if material_eval >= beta {
        return material_eval;
    }

    if material_eval > alpha {
        alpha = material_eval;
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
