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

use shallow_guess::bit_masks::SQUARE_MASKS;
use shallow_guess::chess_position::ChessPosition;
use shallow_guess::def::{BK, BP, CASTLING_FLAG_EMPTY, CHESS_SQUARE_COUNT, NO_SQUARE, WHITE, WK};
use shallow_guess::generated::network_weights::INPUT_LAYER_SIZE;
use shallow_guess::network::{FastNoOpNetwork, Network};
use shallow_guess::util::{read_lines, win_probability_to_centi_pawn_score};

const ONE_SYMBOL: char = 'X';

fn main() {
    let mut args = env::args().into_iter();
    args.next().unwrap();

    let input_file = args.next().unwrap_or_else(|| {
        eprintln!("Usage: reverse_training_data <input_file> [output_file]");
        std::process::exit(1);
    });
    let output_file = args.next().unwrap_or_else(|| {
        let default = format!("{}.reversed", input_file);
        println!("No output file specified, using {}", default);
        default
    });

    reverse_training_data(&input_file, &output_file);
}

fn reverse_training_data(input_file: &str, output_path: &str) {
    let output_file = File::create(output_path).unwrap();
    let mut writer = LineWriter::new(output_file);

    if let Ok(lines) = read_lines(input_file) {
        for (line_num, line_result) in lines.enumerate() {
            let line = line_result.unwrap();
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() != 3 {
                eprintln!("Line {}: expected exactly 2 commas, skipping", line_num + 1);
                continue;
            }

            let encoded = parts[0];
            let result = parts[1];
            let annotated_score = parts[2];

            match (annotated_score.parse::<f32>(), decode_to_fen(encoded)) {
                (Ok(win_probability), Ok(fen)) => {
                    let centipawn_score = win_probability_to_centi_pawn_score(win_probability);

                    let result_str = match result.trim() {
                        "0.0" => "Loss",
                        "0.5" => "Draw",
                        "1.0" => "Win",
                        other_result => panic!("Unexpected result {}", other_result),
                    };

                    writer
                        .write_all(
                            format!("{}  Result={} Score={}\n", fen, result_str, centipawn_score)
                                .as_bytes(),
                        )
                        .unwrap();
                }
                (Err(_), _) => {
                    eprintln!(
                        "Line {}: error parsing annotated win probability '{}'",
                        line_num + 1,
                        annotated_score
                    );
                }
                (_, Err(e)) => {
                    eprintln!("Line {}: error decoding: {}", line_num + 1, e);
                }
            }
        }
    }

    println!("Reversed training data written to {}", output_path);
}

fn decode_to_fen(encoded: &str) -> Result<String, String> {
    let bits = decode_compressed_string(encoded)?;

    let network = FastNoOpNetwork::new();
    let mut position = ChessPosition::new(network);

    position.player = WHITE;
    position.enpassant_square = NO_SQUARE;
    position.castling_flag = CASTLING_FLAG_EMPTY;
    position.half_move_count = 0;
    position.full_move_count = 1;

    for (idx, &bit) in bits.iter().enumerate() {
        if bit == 1 {
            let piece_type = idx / CHESS_SQUARE_COUNT + 1;
            let square = idx % CHESS_SQUARE_COUNT;
            let piece = piece_type as u8;

            position.board[square] = piece;
            position.bitboards[piece as usize] |= SQUARE_MASKS[square];
            position.network.add(piece, square);

            if piece < BP {
                position.white_all_bitboard |= SQUARE_MASKS[square];
                if piece == WK {
                    position.white_king_square = square;
                }
            } else {
                position.black_all_bitboard |= SQUARE_MASKS[square];
                if piece == BK {
                    position.black_king_square = square;
                }
            }
        }
    }

    let fen = position.to_fen();
    Ok(fen)
}

fn decode_compressed_string(encoded: &str) -> Result<Vec<u8>, String> {
    let mut bits = Vec::with_capacity(INPUT_LAYER_SIZE);
    let mut chars = encoded.chars().peekable();
    let mut zero_run = 0;

    while let Some(c) = chars.next() {
        if c == ONE_SYMBOL {
            if zero_run > 0 {
                for _ in 0..zero_run {
                    bits.push(0);
                }
                zero_run = 0;
            }
            bits.push(1);
        } else if c.is_ascii_digit() {
            let mut num_str = String::new();
            num_str.push(c);
            while let Some(&next) = chars.peek() {
                if next.is_ascii_digit() {
                    num_str.push(chars.next().unwrap());
                } else {
                    break;
                }
            }
            let count: usize = num_str
                .parse()
                .map_err(|_| format!("Invalid zero count {}", num_str))?;
            zero_run += count;
        } else {
            return Err(format!("Unexpected character '{}' in encoded string", c));
        }
    }

    if zero_run > 0 {
        for _ in 0..zero_run {
            bits.push(0);
        }
    }

    if bits.len() != INPUT_LAYER_SIZE {
        return Err(format!(
            "Decoded length {} does not match INPUT_LAYER_SIZE {}",
            bits.len(),
            INPUT_LAYER_SIZE
        ));
    }

    Ok(bits)
}
