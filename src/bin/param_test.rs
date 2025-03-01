use shallow_guess::chess_position::ChessPosition;
use shallow_guess::def::STACK_SIZE_BYTES;
use shallow_guess::fen::format_chess_move;
use shallow_guess::network::Network;
use shallow_guess::search_engine::SearchEngine;
use shallow_guess::transpos::{TranspositionTable, DEFAULT_HASH_SIZE_MB};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;
use std::{fs, thread};

const MAX_ALLOWED_SEARCH_TIME_SECS: u64 = 120;

fn main() -> std::io::Result<()> {
    let args = std::env::args().collect::<Vec<String>>();

    let epd_file = &args[1];
    let epd_content = fs::read_to_string(&epd_file)?;

    thread::Builder::new()
        .stack_size(STACK_SIZE_BYTES)
        .spawn(move || {
            let network = Network::new();
            let mut chess_position = ChessPosition::new(network);
            let transposition_table = TranspositionTable::new(DEFAULT_HASH_SIZE_MB);
            let mut search_engine = SearchEngine::new(transposition_table);

            let mut search_time_secs = 10;

            loop {
                println!("Testing under {} seconds timebox...", search_time_secs);

                let mut success_count = 0;
                let mut failure_count = 0;

                for line in epd_content.lines() {
                    let mut test_case = line.split(";");
                    let fen = test_case.next().unwrap();
                    let expected_best_move = test_case.next().unwrap();

                    chess_position.set_from_fen(fen);

                    println!("Testing [{}]", fen);

                    let best_move_str = format_chess_move(&search_engine.search_best_move(
                        &mut chess_position,
                        Duration::from_secs(search_time_secs),
                        Arc::new(AtomicBool::new(false)),
                        true,
                    ));

                    if best_move_str == expected_best_move {
                        success_count += 1;
                        println!("✅");
                    } else {
                        failure_count += 1;

                        println!("❌, expected {}, got {}", expected_best_move, best_move_str);
                    }

                    println!(
                        "Passed: {}/{}",
                        success_count,
                        success_count + failure_count
                    );
                }

                println!(
                    "[{} seconds] {} tests passed, {} tests failed, success rate {}%",
                    search_time_secs,
                    success_count,
                    failure_count,
                    success_count * 100 / (success_count + failure_count)
                );
                println!("----------------------------------------------------");

                if failure_count == 0 || search_time_secs >= MAX_ALLOWED_SEARCH_TIME_SECS {
                    break;
                }

                search_time_secs *= 2;
                search_engine.reset_game();
            }
        })
        .unwrap()
        .join()
        .unwrap();

    Ok(())
}
