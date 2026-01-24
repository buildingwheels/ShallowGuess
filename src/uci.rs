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

use crate::chess_game::ChessGame;
use crate::chess_position::ChessPosition;
use crate::def::{
    BK, BP, C1, C8, CREATE_ENPASSANT_DELTA, E1, E8, G1, G8, MATE_SCORE, NO_PIECE, TERMINATE_SCORE,
    WHITE, WK, WP,
};
use crate::engine_info::{AUTHOR, ENGINE_NAME, ENGINE_VERSION};
use crate::fen::fen_str_constants::{SPLITTER, START_POS};
use crate::fen::{
    format_chess_move, get_chess_square_from_chars, get_promotion_chess_piece_from_char,
};
use crate::generated::network_weights::HIDDEN_LAYER_SIZE;
use crate::search_engine::SearchInfo;
use crate::time::{calculate_optimal_time_for_next_move, TimeInfo};
use crate::transpos::{DEFAULT_HASH_SIZE_MB, MAX_HASH_SIZE_MB, MIN_HASH_SIZE_MB};
use crate::types::{ChessMove, ChessMoveCount, ChessMoveType, MilliSeconds, SearchDepth};
use crate::util::is_power_of_two;

use crate::network::Network;
use std::str::SplitWhitespace;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::vec::Vec;

const UCI_CMD_UCI: &str = "uci";
const UCI_CMD_IS_READY: &str = "isready";
const UCI_CMD_UCI_NEW_GAME: &str = "ucinewgame";
const UCI_CMD_POSITION: &str = "position";
const UCI_CMD_SET_OPTION: &str = "setoption";
const UCI_CMD_GO: &str = "go";
const UCI_CMD_STOP: &str = "stop";
const UCI_CMD_QUIT: &str = "quit";

const UCI_CMD_OPTION_KEY_NAME: &str = "name";
const UCI_CMD_OPTION_KEY_VALUE: &str = "value";
const UCI_CMD_OPTION_VALUE_HASH: &str = "Hash";
const UCI_CMD_MOVES: &str = "moves";
const UCI_CMD_START_POS: &str = "startpos";
const UCI_CMD_FEN: &str = "fen";
const UCI_CMD_PERFT: &str = "perft";
const UCI_CMD_INFINITE: &str = "infinite";
const UCI_CMD_WHITE_TIME: &str = "wtime";
const UCI_CMD_BLACK_TIME: &str = "btime";
const UCI_CMD_MOVETIME: &str = "movetime";
const UCI_CMD_WHITE_TIME_INCREMENT: &str = "winc";
const UCI_CMD_BLACK_TIME_INCREMENT: &str = "binc";
const UCI_CMD_MOVES_TO_GO: &str = "movestogo";

pub fn process_command<N: Network + 'static>(
    command: &str,
    chess_game: Arc<Mutex<ChessGame<N>>>,
    force_stopped: Arc<AtomicBool>,
) {
    let mut subcommands = command.trim().split_whitespace();

    if let Some(first_subcommand) = subcommands.next() {
        match first_subcommand {
            UCI_CMD_UCI => print_engine_info(),
            UCI_CMD_IS_READY => print_ready_message(),
            UCI_CMD_UCI_NEW_GAME => chess_game.lock().unwrap().reset_game(),
            UCI_CMD_POSITION => {
                process_set_position_and_move_list_command(&mut subcommands, chess_game)
            }
            UCI_CMD_SET_OPTION => process_set_option_command(&mut subcommands, chess_game),
            UCI_CMD_GO => {
                process_go_command(&mut subcommands, chess_game, Arc::clone(&force_stopped))
            }
            UCI_CMD_STOP => force_stopped.store(true, Ordering::Relaxed),
            UCI_CMD_QUIT => std::process::exit(0),
            _ => println!("Unknown command: {}", first_subcommand),
        }
    }
}

fn get_target_info() -> String {
    let mut features = Vec::new();

    #[cfg(target_feature = "avx512f")]
    features.push("AVX512");
    #[cfg(target_feature = "avx2")]
    features.push("AVX2");
    #[cfg(target_feature = "avx")]
    features.push("AVX");
    #[cfg(target_feature = "sse4.2")]
    features.push("SSE4.2");
    #[cfg(target_feature = "sse4.1")]
    features.push("SSE4.1");
    #[cfg(target_feature = "ssse3")]
    features.push("SSSE3");
    #[cfg(target_feature = "sse3")]
    features.push("SSE3");
    #[cfg(target_feature = "sse2")]
    features.push("SSE2");
    #[cfg(target_feature = "neon")]
    features.push("NEON");

    if features.is_empty() {
        "Scalar".to_string()
    } else {
        features.join("+")
    }
}

fn print_engine_info() {
    println!(
        "id name {} {} [{}] [{}] [{}]",
        ENGINE_NAME,
        ENGINE_VERSION,
        HIDDEN_LAYER_SIZE,
        std::env::consts::ARCH,
        get_target_info()
    );
    println!("id author {}", AUTHOR);
    println!(
        "option name Hash type spin default {} min {} max {}",
        DEFAULT_HASH_SIZE_MB, MIN_HASH_SIZE_MB, MAX_HASH_SIZE_MB
    );
    println!("uciok");
}

fn print_ready_message() {
    println!("readyok");
}

#[inline(always)]
fn print_best_move(chess_move: &ChessMove) {
    println!("bestmove {}", format_chess_move(chess_move));
}

pub fn print_info(search_info: SearchInfo, principal_variation: &[ChessMove]) {
    let score = search_info.score;
    let abs_score = score.abs();
    let display_score = if abs_score > TERMINATE_SCORE {
        let mate_distance = (MATE_SCORE - abs_score + 1) / 2;

        if score > 0 {
            format!("mate {}", mate_distance)
        } else {
            format!("mate {}", -mate_distance)
        }
    } else {
        format!("cp {}", score)
    };

    print!(
        "info score {} nodes {} time {} depth {} seldepth {} hashfull {} pv",
        display_score,
        search_info.searched_node_count,
        search_info.searched_time_ms,
        search_info.depth,
        search_info.selected_depth,
        search_info.hash_utilization_permil,
    );

    for chess_move in principal_variation {
        print!(" {}", format_chess_move(chess_move));
    }

    println!();
}

fn process_set_position_and_move_list_command<N: Network + 'static>(
    subcommands: &mut SplitWhitespace,
    chess_game: Arc<Mutex<ChessGame<N>>>,
) {
    let next_command = subcommands
        .next()
        .expect("Value needs to be provided after position");

    match next_command {
        UCI_CMD_START_POS => {
            chess_game.lock().unwrap().set_position_from_fen(START_POS);
            if let Some(UCI_CMD_MOVES) = subcommands.next() {
                process_moves(subcommands, chess_game);
            }
        }
        UCI_CMD_FEN => {
            let mut fen_parts = Vec::new();
            for part in subcommands.by_ref() {
                if part == UCI_CMD_MOVES {
                    break;
                }
                fen_parts.push(part);
            }
            let fen_string = fen_parts.join(SPLITTER);

            chess_game
                .lock()
                .unwrap()
                .set_position_from_fen(&fen_string);
            process_moves(subcommands, chess_game);
        }
        _ => println!(
            "Unsupported or invalid command after position: {}",
            next_command
        ),
    }
}

fn process_moves<N: Network + 'static>(
    subcommands: &mut SplitWhitespace,
    chess_game: Arc<Mutex<ChessGame<N>>>,
) {
    let mut game_lock = chess_game.lock().unwrap();
    while let Some(next_chess_move_str) = subcommands.next() {
        let next_chess_move = parse_move_str(next_chess_move_str, game_lock.get_position());
        game_lock.make_move(&next_chess_move);
    }
}

fn process_set_option_command<N: Network + 'static>(
    subcommands: &mut SplitWhitespace,
    chess_game: Arc<Mutex<ChessGame<N>>>,
) {
    while let Some(next_command) = subcommands.next() {
        match next_command {
            UCI_CMD_OPTION_KEY_NAME => {
                if let Some(option_name) = subcommands.next() {
                    if option_name != UCI_CMD_OPTION_VALUE_HASH {
                        println!("Unsupported option {}", option_name);
                        break;
                    }
                }
            }
            UCI_CMD_OPTION_KEY_VALUE => {
                if let Some(hash_size_str) = subcommands.next() {
                    if let Ok(hash_size_mb) = hash_size_str.parse::<usize>() {
                        if !is_power_of_two(hash_size_mb) {
                            println!("Invalid hash size {}, needs to be power of 2", hash_size_mb);
                        } else {
                            chess_game.lock().unwrap().set_hash_size(hash_size_mb);
                        }
                    } else {
                        println!("Unable to parse hash size: {}", hash_size_str);
                    }
                }
            }
            _ => {}
        }
    }
}

fn process_go_command<N: Network + 'static>(
    subcommands: &mut SplitWhitespace,
    chess_game: Arc<Mutex<ChessGame<N>>>,
    force_stopped: Arc<AtomicBool>,
) {
    let mut white_time = TimeInfo::new();
    let mut black_time = TimeInfo::new();

    while let Some(next_command) = subcommands.next() {
        match next_command {
            UCI_CMD_PERFT => {
                process_perft_command(subcommands, chess_game);
                return;
            }
            UCI_CMD_INFINITE => {
                let infinite_time = MilliSeconds::MAX;
                white_time.remaining_time_millis = infinite_time;
                white_time.remaining_move_count = 1;
                black_time.remaining_time_millis = infinite_time;
                black_time.remaining_move_count = 1;
            }
            UCI_CMD_MOVETIME => {
                if let Some(after_command) = subcommands.next() {
                    if let Ok(time) = after_command.parse::<MilliSeconds>() {
                        white_time.remaining_time_millis = time;
                        white_time.remaining_move_count = 1;
                        black_time.remaining_time_millis = time;
                        black_time.remaining_move_count = 1;
                    } else {
                        println!("Unable to parse move time: {}", after_command);
                    }
                } else {
                    println!("Value needs to be provided after movetime");
                }
            }
            UCI_CMD_WHITE_TIME | UCI_CMD_WHITE_TIME_INCREMENT => {
                process_time_command(subcommands, &mut white_time, next_command);
            }
            UCI_CMD_BLACK_TIME | UCI_CMD_BLACK_TIME_INCREMENT => {
                process_time_command(subcommands, &mut black_time, next_command);
            }
            UCI_CMD_MOVES_TO_GO => {
                if let Some(after_command) = subcommands.next() {
                    if let Ok(remaining_move_count) = after_command.parse::<ChessMoveCount>() {
                        white_time.remaining_move_count = remaining_move_count;
                        black_time.remaining_move_count = remaining_move_count;
                    } else {
                        println!("Unable to parse movestogo: {}", after_command);
                        break;
                    }
                } else {
                    println!("Value needs to be provided after movestogo");
                    break;
                }
            }
            unknown_next_command => {
                println!("Unknown sub command: {}", unknown_next_command);
                break;
            }
        }
    }

    let chess_game = Arc::clone(&chess_game);
    let force_stopped_clone = Arc::clone(&force_stopped);
    force_stopped.store(false, Ordering::Relaxed);

    thread::spawn(move || {
        let (player, debug_info) = {
            let game_lock = chess_game.lock().unwrap();
            (game_lock.get_position().player, game_lock.get_debug_info())
        };

        let (allowed_time_ms, extra_time_ms) = if player == WHITE {
            calculate_optimal_time_for_next_move(&white_time)
        } else {
            calculate_optimal_time_for_next_move(&black_time)
        };

        let best_move = {
            let mut chess_game = chess_game.lock().unwrap();
            chess_game.search_best_move(allowed_time_ms, extra_time_ms, force_stopped_clone)
        };

        if best_move.is_empty() {
            println!("Unable to find chess move for {}", debug_info);
        } else {
            print_best_move(&best_move);
        }
    });
}

fn process_time_command(
    subcommands: &mut SplitWhitespace,
    time_info: &mut TimeInfo,
    command: &str,
) {
    if let Some(after_command) = subcommands.next() {
        if let Ok(time) = after_command.parse::<MilliSeconds>() {
            if command == UCI_CMD_WHITE_TIME || command == UCI_CMD_BLACK_TIME {
                time_info.remaining_time_millis = time;
            } else {
                time_info.increment_time_millis = time;
            }
        } else {
            println!("Unable to parse {}: {}", command, after_command);
        }
    } else {
        println!("Value needs to be provided after {}", command);
    }
}

fn process_perft_command<N: Network + 'static>(
    subcommands: &mut SplitWhitespace,
    chess_game: Arc<Mutex<ChessGame<N>>>,
) {
    if let Some(depth_str) = subcommands.next() {
        if let Ok(depth) = depth_str.parse::<SearchDepth>() {
            chess_game.lock().unwrap().perft(depth);
        } else {
            println!("Unable to parse provided depth: {}", depth_str);
        }
    } else {
        println!("Depth value needs to be provided after perft");
    }
}

fn parse_move_str<N: Network + 'static>(
    move_str: &str,
    chess_position: &ChessPosition<N>,
) -> ChessMove {
    let mut move_str_iter = move_str.chars();
    let from_file = move_str_iter.next().expect("Invalid move format");
    let from_rank = move_str_iter.next().expect("Invalid move format");
    let to_file = move_str_iter.next().expect("Invalid move format");
    let to_rank = move_str_iter.next().expect("Invalid move format");

    let from_square = get_chess_square_from_chars(from_file, from_rank);
    let to_square = get_chess_square_from_chars(to_file, to_rank);

    let mut move_type = ChessMoveType::Regular;
    let mut promotion_piece = NO_PIECE;

    let moving_piece = chess_position.board[from_square];

    if move_str.len() > 4 {
        move_type = ChessMoveType::Promotion;
        promotion_piece = get_promotion_chess_piece_from_char(
            move_str_iter.next().expect("Invalid promotion piece"),
            chess_position.player,
        );
    } else if moving_piece == WK && from_square == E1 && (to_square == G1 || to_square == C1) {
        move_type = ChessMoveType::Castle;
    } else if moving_piece == BK && from_square == E8 && (to_square == G8 || to_square == C8) {
        move_type = ChessMoveType::Castle;
    } else if moving_piece == WP {
        if to_square == chess_position.enpassant_square {
            move_type = ChessMoveType::EnPassant;
        } else if to_square - from_square == CREATE_ENPASSANT_DELTA {
            move_type = ChessMoveType::CreateEnPassant;
        }
    } else if moving_piece == BP {
        if to_square == chess_position.enpassant_square {
            move_type = ChessMoveType::EnPassant;
        } else if from_square - to_square == CREATE_ENPASSANT_DELTA {
            move_type = ChessMoveType::CreateEnPassant;
        }
    }

    ChessMove {
        from_square,
        to_square,
        promotion_piece,
        move_type,
    }
}
