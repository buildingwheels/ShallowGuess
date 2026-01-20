// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

use shallow_guess::chess_move_gen::is_in_check;
use shallow_guess::chess_position::ChessPosition;
use shallow_guess::def::{DRAW_SCORE, STACK_SIZE_BYTES, TERMINATE_SCORE};
use shallow_guess::network::QuantizedNetwork;
use shallow_guess::search_engine::SearchEngine;
use shallow_guess::transpos::{TranspositionTable, DEFAULT_HASH_SIZE_MB};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::thread;

fn main() -> std::io::Result<()> {
    let args = std::env::args().collect::<Vec<String>>();
    if args.len() != 3 {
        eprintln!("Usage: {} <fen_file_path> <stats_batch_count>", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];
    let fen_positions = read_fen_positions(file_path);
    println!("Loaded {} FEN positions", fen_positions.len());

    let stats_batch_count: usize = args[2].parse().expect("Invalid stats batch count");

    thread::Builder::new()
        .stack_size(STACK_SIZE_BYTES)
        .spawn(move || {
            let network = QuantizedNetwork::new();
            let mut chess_position = ChessPosition::new(network);
            let transposition_table = TranspositionTable::new(DEFAULT_HASH_SIZE_MB);
            let mut search_engine = SearchEngine::new(transposition_table);

            let mut potential_drops_stats = Vec::new();
            let mut processed_position_count = 0;
            let mut skipped_in_check_position_count = 0;
            let mut skipped_high_static_position_count = 0;
            let mut skipped_high_search_score_position_count = 0;

            for (i, fen) in fen_positions.iter().enumerate() {
                chess_position.set_from_fen(fen);

                if is_in_check(&chess_position) {
                    skipped_in_check_position_count += 1;
                    continue;
                }

                let static_score = chess_position.get_static_score();

                if static_score >= TERMINATE_SCORE || static_score <= -TERMINATE_SCORE {
                    skipped_high_static_position_count += 1;
                    continue;
                }

                let search_score = search_engine.search_to_depth(&mut chess_position, 1);

                if search_score == DRAW_SCORE
                    || search_score >= TERMINATE_SCORE
                    || search_score <= -TERMINATE_SCORE
                {
                    skipped_high_search_score_position_count += 1;
                    continue;
                }

                if search_score < static_score {
                    let diff = static_score - search_score;
                    potential_drops_stats.push(diff);
                }

                processed_position_count += 1;
                search_engine.reset_game();

                if i % stats_batch_count == 0 {
                    println!("Summary after {} positions", i);
                    println!();

                    println!("Potential Drops:");
                    print_statistics(&potential_drops_stats);
                    println!();
                }
            }

            println!("Final Summary after {} positions ({} skipped due to in check, {} skipped due to high static score, {} skipped due to high search score",
                processed_position_count, skipped_in_check_position_count, skipped_high_static_position_count, skipped_high_search_score_position_count);
            println!();

            println!("Potential Drops:");
            print_statistics(&potential_drops_stats);
        })
        .unwrap()
        .join()
        .unwrap();

    Ok(())
}

fn read_fen_positions(file_path: &str) -> Vec<String> {
    let file = File::open(file_path).expect("Failed to open FEN file");
    let reader = BufReader::new(file);

    let mut fens = Vec::new();

    for line in reader.lines() {
        if let Ok(line) = line {
            let trimmed_line = line.trim();
            if trimmed_line.is_empty() {
                continue;
            }

            let fen = trimmed_line;
            let processed_fen = fen[..fen.find(',').unwrap_or(fen.len())].trim().to_string();
            fens.push(processed_fen);
        }
    }

    println!("Processed {} FEN positions", fens.len());
    fens
}

fn print_statistics(stats: &Vec<i32>) {
    if stats.is_empty() {
        return;
    }

    let mut sorted_stats = stats.clone();
    sorted_stats.sort();

    let max_val = sorted_stats[sorted_stats.len() - 1];

    let p95_idx = ((stats.len() as f64 * 0.95).floor() as usize).saturating_sub(1);
    let p99_idx = ((stats.len() as f64 * 0.99).floor() as usize).saturating_sub(1);
    let p99_9_idx = ((stats.len() as f64 * 0.999).floor() as usize).saturating_sub(1);
    let p99_99_idx = ((stats.len() as f64 * 0.9999).floor() as usize).saturating_sub(1);

    let p95 = sorted_stats[p95_idx] as f64;
    let p99 = sorted_stats[p99_idx] as f64;
    let p99_9 = sorted_stats[p99_9_idx] as f64;
    let p99_99 = sorted_stats[p99_99_idx] as f64;

    println!(
        "max={}, p99.99={:.2}, p99.9={:.2}, p99={:.2}, p95={:.2}",
        max_val, p99_99, p99_9, p99, p95
    );
}
