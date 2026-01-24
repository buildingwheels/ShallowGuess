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

use shallow_guess::types::ChessMoveCount;
use shallow_guess::util::read_lines;
use std::env;
use std::fs::File;
use std::io::{LineWriter, Write};

fn main() {
    let mut args = env::args().into_iter();
    args.next().unwrap();

    let original_fen_file = args.next().unwrap();
    let output_file = args.next().unwrap();
    let skip_count = args.next().unwrap().parse::<ChessMoveCount>().unwrap();
    let max_count = args.next().unwrap().parse::<ChessMoveCount>().unwrap();
    let skip_terminated_game = args.next().unwrap().parse::<bool>().unwrap();

    filter_fen_data(
        &original_fen_file,
        &output_file,
        skip_count,
        max_count,
        skip_terminated_game,
    );
}

fn filter_fen_data(
    fen_file: &str,
    output_file_path: &str,
    skip_count: ChessMoveCount,
    max_count: ChessMoveCount,
    skip_terminated_game: bool,
) {
    let output_file = File::create(output_file_path).unwrap();
    let mut file_writer = LineWriter::new(output_file);

    let mut total_position_count: u128 = 0;

    let mut current_game_result = 0.;
    let mut current_game_position_count = 0;
    let mut skip_current_game = false;

    if let Ok(lines) = read_lines(fen_file) {
        for line in lines.flatten() {
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            if skip_terminated_game && line.contains("Termination") {
                skip_current_game = true;
            } else if line.contains("Result") {
                if line.contains("1/2") {
                    current_game_result = 0.5;
                } else if line.contains("0-1") {
                    current_game_result = 0.;
                } else if line.contains("1-0") {
                    current_game_result = 1.;
                }

                current_game_position_count = 0;
                skip_current_game = false;
            } else if !line.contains("[") {
                if skip_current_game {
                    continue;
                }

                current_game_position_count += 1;
                total_position_count += 1;

                if current_game_position_count > skip_count
                    && current_game_position_count < max_count
                {
                    file_writer
                        .write(format!("{},{}\n", line.trim(), current_game_result,).as_bytes())
                        .unwrap();
                }
            }
        }
    }

    println!(
        "Completed filtering FEN data with size {} to: {}",
        total_position_count, output_file_path
    );
}
