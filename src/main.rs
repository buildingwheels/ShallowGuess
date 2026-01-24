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

use shallow_guess::chess_game::ChessGame;
use shallow_guess::chess_position::ChessPosition;
use shallow_guess::def::STACK_SIZE_BYTES;
use shallow_guess::network::QuantizedNetwork;
use shallow_guess::search_engine::SearchEngine;
use shallow_guess::transpos::{TranspositionTable, DEFAULT_HASH_SIZE_MB};
use shallow_guess::uci::process_command;
use std::io::{stdin, BufRead};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    thread::Builder::new()
        .stack_size(STACK_SIZE_BYTES)
        .spawn(|| {
            run_uci_game();
        })
        .unwrap()
        .join()
        .unwrap();
}

fn run_uci_game() {
    let chess_position = ChessPosition::new(QuantizedNetwork::new());
    let transposition_table = TranspositionTable::new(DEFAULT_HASH_SIZE_MB);
    let search_engine = SearchEngine::new(transposition_table);
    let chess_game = Arc::new(Mutex::new(ChessGame::new(chess_position, search_engine)));
    let force_stopped = Arc::new(AtomicBool::new(false));

    loop {
        let mut input = String::new();

        match stdin().lock().read_line(&mut input) {
            Ok(_) => {}
            Err(error) => eprintln!("unable to read input {}", error),
        }

        process_command(&input, Arc::clone(&chess_game), Arc::clone(&force_stopped));
    }
}
