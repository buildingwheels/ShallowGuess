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

use std::io::{self, BufRead, Write};

use shallow_guess::chess_position::ChessPosition;
use shallow_guess::def::{BLACK, WHITE};
use shallow_guess::network::FastNoOpNetwork;

const RESET: &str = "\x1b[0m";
const WHITE_PIECE: &str = "\x1b[32m";
const BLACK_PIECE: &str = "\x1b[31m";
const THICK_GREEN_LINE: &str = "\x1b[32m═══════════════════════════════════════\x1b[0m";
const THICK_RED_LINE: &str = "\x1b[31m═══════════════════════════════════════\x1b[0m";

fn piece_to_unicode(piece: u8) -> &'static str {
    match piece {
        1 => "♟",
        2 => "♞",
        3 => "♝",
        4 => "♜",
        5 => "♛",
        6 => "♚",
        7 => "♟",
        8 => "♞",
        9 => "♝",
        10 => "♜",
        11 => "♛",
        12 => "♚",
        _ => " ",
    }
}

fn piece_color(piece: u8) -> &'static str {
    if piece == 0 {
        return "";
    }
    if piece <= 6 {
        WHITE_PIECE
    } else {
        BLACK_PIECE
    }
}

fn print_board(chess_position: &ChessPosition<FastNoOpNetwork>) {
    println!();

    if chess_position.player == BLACK {
        println!("{}", THICK_RED_LINE);
    }

    println!("  +---+---+---+---+---+---+---+---+");

    for rank in (0..8).rev() {
        print!("{} |", rank + 1);

        for file in 0..8 {
            let square_index = rank * 8 + file;
            let piece = chess_position.board[square_index];
            let piece_str = piece_to_unicode(piece);
            let color = piece_color(piece);

            print!(" {}{}{} |", color, piece_str, RESET);
        }

        println!();
        println!("  +---+---+---+---+---+---+---+---+");
    }

    println!("    a   b   c   d   e   f   g   h");

    if chess_position.player == WHITE {
        println!("{}", THICK_GREEN_LINE);
    }

    println!();
}

fn main() {
    let stdin = io::stdin();
    let mut chess_position = ChessPosition::new(FastNoOpNetwork::new());
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        match line {
            Ok(input) => {
                let fen = input.trim();

                if fen.is_empty() {
                    continue;
                }

                chess_position.set_from_fen(fen);
                print_board(&chess_position);
                println!();

                let _ = stdout.flush();
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
    }
}
