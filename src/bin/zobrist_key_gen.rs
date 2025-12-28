// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

use shallow_guess::def::{
    A8, BLACK, CASTLING_FLAG_BLACK_KING_SIDE, CASTLING_FLAG_BLACK_QUEEN_SIDE, CASTLING_FLAG_EMPTY,
    CASTLING_FLAG_FULL, CASTLING_FLAG_WHITE_KING_SIDE, CASTLING_FLAG_WHITE_QUEEN_SIDE,
    CHESS_FILE_COUNT, CHESS_SQUARE_COUNT, PIECE_TYPE_COUNT, PLAYER_COUNT, WHITE,
};
use shallow_guess::fen::{
    fen_str_constants, get_chess_piece_from_char, get_chess_square_from_chars,
};
use shallow_guess::prng::RandGenerator;
use shallow_guess::types::{CastlingFlag, ChessPiece, ChessSquare, HashKey, Player};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <fen_file_path> <max_seeds_count>", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];
    let max_seeds_count = &args[2].parse::<usize>().unwrap();

    let fen_positions = read_fen_positions(file_path);
    println!("Loaded {} unique FEN positions", fen_positions.len());

    let best_seed = find_best_seed(&fen_positions, *max_seeds_count);

    println!(
        "Best seed: {} with duplicate rate: {:.12}%, collision rate: {:.12}%",
        best_seed.0,
        best_seed.1 * 100.0,
        best_seed.2 * 100.0
    );

    let tables = ZobristTables::new(best_seed.0);
    output_zobrist_tables(&tables, best_seed.0);
}

#[derive(Clone)]
pub struct ZobristTables {
    pub piece_square_hash: [[HashKey; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT],
    pub castling_flag_hash: [HashKey; CASTLING_FLAG_FULL as usize + 1],
    pub enpassant_square_hash: [HashKey; CHESS_SQUARE_COUNT],
    pub player_hash: [HashKey; PLAYER_COUNT],
}

impl ZobristTables {
    pub fn new(seed: u64) -> Self {
        let mut rng = RandGenerator::new(seed);
        let mut used_keys = std::collections::HashSet::new();

        let mut piece_square_hash = [[0; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];
        for piece in 0..PIECE_TYPE_COUNT {
            for square in 0..CHESS_SQUARE_COUNT {
                let mut key;
                loop {
                    key = rng.next() as HashKey;
                    if !used_keys.contains(&key) {
                        used_keys.insert(key);
                        break;
                    }
                }
                piece_square_hash[piece][square] = key;
            }
        }

        let mut castling_flag_hash = [0; CASTLING_FLAG_FULL as usize + 1];
        for flag in 0..=CASTLING_FLAG_FULL as usize {
            let mut key;
            loop {
                key = rng.next() as HashKey;
                if !used_keys.contains(&key) {
                    used_keys.insert(key);
                    break;
                }
            }
            castling_flag_hash[flag] = key;
        }

        let mut enpassant_square_hash = [0; CHESS_SQUARE_COUNT];
        for square in 0..CHESS_SQUARE_COUNT {
            let mut key;
            loop {
                key = rng.next() as HashKey;
                if !used_keys.contains(&key) {
                    used_keys.insert(key);
                    break;
                }
            }
            enpassant_square_hash[square] = key;
        }

        let mut player_hash = [0; PLAYER_COUNT];
        for player in 0..PLAYER_COUNT {
            let mut key;
            loop {
                key = rng.next() as HashKey;
                if !used_keys.contains(&key) {
                    used_keys.insert(key);
                    break;
                }
            }
            player_hash[player] = key;
        }

        ZobristTables {
            piece_square_hash,
            castling_flag_hash,
            enpassant_square_hash,
            player_hash,
        }
    }
}

#[derive(Clone)]
pub struct ZobristHasher {
    hash: HashKey,
    tables: ZobristTables,
}

impl ZobristHasher {
    pub fn from_tables(tables: ZobristTables) -> Self {
        ZobristHasher { hash: 0, tables }
    }

    pub fn get_hash(&self) -> HashKey {
        self.hash
    }

    pub fn reset(&mut self) {
        self.hash = 0;
    }

    pub fn toggle_piece(&mut self, piece: ChessPiece, square: ChessSquare) {
        self.hash ^= self.tables.piece_square_hash[piece as usize][square];
    }

    pub fn toggle_castling_flag(&mut self, flag: CastlingFlag) {
        self.hash ^= self.tables.castling_flag_hash[flag as usize];
    }

    pub fn toggle_enpassant_square(&mut self, square: ChessSquare) {
        self.hash ^= self.tables.enpassant_square_hash[square];
    }

    pub fn toggle_player(&mut self, player: Player) {
        self.hash ^= self.tables.player_hash[player as usize];
    }
}

fn read_fen_positions(file_path: &str) -> Vec<String> {
    let file = File::open(file_path).expect("Failed to open FEN file");
    let reader = BufReader::new(file);

    let mut unique_fens = std::collections::HashSet::new();
    let mut total_count = 0;

    for line in reader.lines() {
        if let Ok(line) = line {
            let trimmed_line = line.trim();
            if trimmed_line.is_empty() {
                continue;
            }

            total_count += 1;
            let fen = trimmed_line;
            let processed_fen = fen[..fen.find(',').unwrap_or(fen.len())].trim().to_string();

            unique_fens.insert(processed_fen);
        }
    }

    let result: Vec<String> = unique_fens.into_iter().collect();

    println!(
        "Processed {} FEN positions, removed {} duplicates",
        total_count,
        total_count - result.len()
    );

    result
}

fn find_best_seed(fen_positions: &[String], num_seeds: usize) -> (u64, f64, f64) {
    let mut best_seed = (0, 1.0, 1.0);
    let table_size = 1024 * 1024 * 16384;

    let mut current_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    for _ in 0..num_seeds {
        let mut rng = RandGenerator::new(current_seed);
        let rand_val = rng.next();

        let next_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let seed = current_seed
            .wrapping_add(rand_val)
            .wrapping_add(next_timestamp);

        let tables = ZobristTables::new(seed);
        let duplicate_rate = calculate_duplicate_rate(&tables, fen_positions);
        let collision_rate = calculate_collision_rate(&tables, fen_positions, table_size);

        println!(
            "Testing seed: {} with duplicate rate: {:.12}%, collision rate: {:.12}%",
            seed,
            duplicate_rate * 100.0,
            collision_rate * 100.0
        );

        let should_update = duplicate_rate < best_seed.1
            || (duplicate_rate == best_seed.1 && collision_rate < best_seed.2);

        if should_update {
            best_seed = (seed, duplicate_rate, collision_rate);
            println!(
                "New best seed: {} with duplicate rate: {:.12}%, collision rate: {:.12}%",
                seed,
                duplicate_rate * 100.0,
                collision_rate * 100.0
            );
        } else {
            println!(
                "Rates: {:.12}% duplicate, {:.12}% collision are not better than current best {:.12}% duplicate, {:.12}% collision",
                duplicate_rate * 100.0,
                collision_rate * 100.0,
                best_seed.1 * 100.0,
                best_seed.2 * 100.0
            );
        }

        current_seed = seed
            .wrapping_add(rand_val)
            .wrapping_add(next_timestamp >> 32);

        if duplicate_rate == 0.0 {
            println!("Perfect seed found: {}", seed);
            break;
        }
    }

    best_seed
}

fn calculate_duplicate_rate(tables: &ZobristTables, fen_positions: &[String]) -> f64 {
    let mut hash_counts = HashMap::new();

    let mut duplicates = 0;
    let mut zero_hash_count = 0;

    for fen in fen_positions {
        let mut hasher = ZobristHasher::from_tables(tables.clone());
        initialize_from_fen(&mut hasher, fen);
        let hash = hasher.get_hash();

        if hash == 0 {
            zero_hash_count += 1;
        }

        let count = hash_counts.entry(hash).or_insert(0);
        *count += 1;

        if *count == 2 {
            duplicates += 1;
        }
    }

    if fen_positions.is_empty() {
        0.0
    } else {
        let duplicate_rate = duplicates as f64 / fen_positions.len() as f64;
        let zero_hash_percentage = (zero_hash_count as f64 / fen_positions.len() as f64) * 100.0;
        eprintln!("Zero hash percentage: {:.4}%", zero_hash_percentage);
        duplicate_rate
    }
}

fn calculate_collision_rate(
    tables: &ZobristTables,
    fen_positions: &[String],
    table_size: usize,
) -> f64 {
    let mut slot_counts = HashMap::new();

    let mask = table_size - 1;

    let mut collisions = 0;

    for fen in fen_positions {
        let mut hasher = ZobristHasher::from_tables(tables.clone());
        initialize_from_fen(&mut hasher, fen);
        let hash = hasher.get_hash();

        let slot = (hash as usize) & mask;

        let count = slot_counts.entry(slot).or_insert(0);
        *count += 1;

        if *count == 2 {
            collisions += 1;
        }
    }

    if fen_positions.is_empty() {
        0.0
    } else {
        collisions as f64 / fen_positions.len() as f64
    }
}

fn initialize_from_fen(hasher: &mut ZobristHasher, fen: &str) {
    hasher.reset();

    let fen = fen.trim();

    let mut fen_segments = fen.split(fen_str_constants::SPLITTER);
    let squares_str = fen_segments.next().unwrap();
    let player_str = fen_segments.next().unwrap();
    let castling_flag_str = fen_segments.next().unwrap();
    let enp_sqr_str = fen_segments.next().unwrap();

    let player = if player_str == fen_str_constants::PLAYER_WHITE {
        WHITE
    } else {
        BLACK
    };
    hasher.toggle_player(player);

    let mut castling_flag = CASTLING_FLAG_EMPTY;
    if castling_flag_str.contains(fen_str_constants::CASTLING_FLAG_WHITE_KING_SIDE) {
        castling_flag |= CASTLING_FLAG_WHITE_KING_SIDE;
    }
    if castling_flag_str.contains(fen_str_constants::CASTLING_FLAG_WHITE_QUEEN_SIDE) {
        castling_flag |= CASTLING_FLAG_WHITE_QUEEN_SIDE;
    }
    if castling_flag_str.contains(fen_str_constants::CASTLING_FLAG_BLACK_KING_SIDE) {
        castling_flag |= CASTLING_FLAG_BLACK_KING_SIDE;
    }
    if castling_flag_str.contains(fen_str_constants::CASTLING_FLAG_BLACK_QUEEN_SIDE) {
        castling_flag |= CASTLING_FLAG_BLACK_QUEEN_SIDE;
    }
    hasher.toggle_castling_flag(castling_flag);

    if enp_sqr_str != fen_str_constants::NA_STR {
        let mut enp_sqr_str_iter = enp_sqr_str.chars().into_iter();
        let enpassant_square = get_chess_square_from_chars(
            enp_sqr_str_iter.next().unwrap(),
            enp_sqr_str_iter.next().unwrap(),
        );
        hasher.toggle_enpassant_square(enpassant_square);
    }

    let mut square_chars = squares_str.chars();
    let mut current_square = A8;

    loop {
        let next_char_option = square_chars.next();

        if next_char_option.is_none() {
            break;
        }

        let next_char = next_char_option.unwrap();

        if next_char == fen_str_constants::RANK_BREAK {
            current_square -= CHESS_FILE_COUNT * 2;
            continue;
        }

        if next_char.is_digit(10) {
            current_square += next_char.to_digit(10).unwrap() as ChessSquare;
            continue;
        }

        let piece = get_chess_piece_from_char(next_char);
        hasher.toggle_piece(piece, current_square);
        current_square += 1;
    }
}

fn output_zobrist_tables(tables: &ZobristTables, seed: u64) {
    let mut output = String::new();

    output.push_str("use crate::def::{CASTLING_FLAG_FULL, CHESS_SQUARE_COUNT, PIECE_TYPE_COUNT, PLAYER_COUNT};\n");
    output.push_str("use crate::types::HashKey;\n\n");

    output.push_str(
        "pub const PIECE_SQUARE_HASH: [[HashKey; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT] = [\n",
    );
    for piece in 0..PIECE_TYPE_COUNT {
        output.push_str("    [\n        ");
        for (i, &key) in tables.piece_square_hash[piece].iter().enumerate() {
            output.push_str(&format!("{}, ", key));
            if (i + 1) % 10 == 0 && i != CHESS_SQUARE_COUNT - 1 {
                output.push_str("\n        ");
            }
        }
        output.push_str("\n    ],\n");
    }
    output.push_str("];\n\n");

    output.push_str(
        "pub const CASTLING_FLAG_HASH: [HashKey; CASTLING_FLAG_FULL as usize + 1] = [\n    ",
    );
    for (i, &key) in tables.castling_flag_hash.iter().enumerate() {
        output.push_str(&format!("{}, ", key));
        if (i + 1) % 8 == 0 && i != tables.castling_flag_hash.len() - 1 {
            output.push_str("\n    ");
        }
    }
    output.push_str("\n];\n\n");

    output.push_str("pub const ENPASSANT_SQUARE_HASH: [HashKey; CHESS_SQUARE_COUNT] = [\n    ");
    for (i, &key) in tables.enpassant_square_hash.iter().enumerate() {
        output.push_str(&format!("{}, ", key));
        if (i + 1) % 8 == 0 && i != CHESS_SQUARE_COUNT - 1 {
            output.push_str("\n    ");
        }
    }
    output.push_str("\n];\n\n");

    output.push_str("pub const PLAYER_HASH: [HashKey; PLAYER_COUNT] = [");
    for &key in tables.player_hash.iter() {
        output.push_str(&format!("{}, ", key));
    }
    output.push_str("];\n\n");

    let output_path = "src/zobrist.rs";
    let mut file = File::create(output_path).expect("Failed to create zobrist.rs file");
    file.write_all(output.as_bytes())
        .expect("Failed to write to zobrist.rs");

    println!(
        "Generated Zobrist tables written to {} with seed: {}",
        output_path, seed
    );
}
