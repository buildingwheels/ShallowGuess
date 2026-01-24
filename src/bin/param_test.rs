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

use shallow_guess::chess_position::ChessPosition;
use shallow_guess::def::STACK_SIZE_BYTES;
use shallow_guess::fen::format_chess_move;
use shallow_guess::network::QuantizedNetwork;
use shallow_guess::search_engine::SearchEngine;
use shallow_guess::transpos::{TranspositionTable, DEFAULT_HASH_SIZE_MB};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;
use std::{fs, thread};

fn main() -> std::io::Result<()> {
    let args = std::env::args().collect::<Vec<String>>();

    let epd_file = &args[1];
    let search_time_secs = args[2].parse::<u64>().unwrap();

    let repeat_search_count = if args.len() > 3 {
        args[3].parse::<u8>().unwrap()
    } else {
        1
    };

    let epd_content = fs::read_to_string(&epd_file)?;

    thread::Builder::new()
        .stack_size(STACK_SIZE_BYTES)
        .spawn(move || {
            let network = QuantizedNetwork::new();
            let mut chess_position = ChessPosition::new(network);
            let transposition_table = TranspositionTable::new(DEFAULT_HASH_SIZE_MB);
            let mut search_engine = SearchEngine::new(transposition_table);

            println!("Testing under {} seconds timebox...", search_time_secs);

            let mut success_count = 0;
            let mut failure_count = 0;

            for line in epd_content.lines() {
                let mut test_case = line.split(";");
                let fen = test_case.next().unwrap();
                let expected_best_move = test_case.next().unwrap();

                chess_position.set_from_fen(fen);

                println!("Testing [{}]", fen);

                for repeat_count in 0..repeat_search_count {
                    println!("Searching iteration {}", repeat_count);

                    let best_move = &search_engine.search_best_move(
                        &mut chess_position,
                        Duration::from_secs(search_time_secs),
                        Duration::from_secs(search_time_secs),
                        None,
                        Arc::new(AtomicBool::new(false)),
                        true,
                    );

                    let best_move_str = format_chess_move(best_move);

                    if best_move_str == expected_best_move {
                        success_count += 1;
                        println!("✅");
                    } else {
                        failure_count += 1;

                        println!("❌, expected {}, got {}", expected_best_move, best_move_str);
                    }
                }

                println!(
                    "Passed: {}/{}",
                    success_count,
                    success_count + failure_count
                );

                search_engine.reset_game();
            }

            println!(
                "[{} seconds] {} tests passed, {} tests failed, success rate {}%",
                search_time_secs,
                success_count,
                failure_count,
                success_count * 100 / (success_count + failure_count)
            );
        })
        .unwrap()
        .join()
        .unwrap();

    Ok(())
}
