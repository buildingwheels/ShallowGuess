// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

use crate::chess_position::ChessPosition;
use crate::fen::fen_str_constants::START_POS;
use crate::network::Network;
use crate::search_engine::SearchEngine;
use crate::types::{ChessMove, MilliSeconds, SearchDepth};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct ChessGame<N: Network> {
    chess_position: ChessPosition<N>,
    search_engine: SearchEngine,
}

impl<N: Network> ChessGame<N> {
    pub fn new(chess_position: ChessPosition<N>, search_engine: SearchEngine) -> Self {
        ChessGame {
            chess_position,
            search_engine,
        }
    }

    pub fn reset_game(&mut self) {
        self.chess_position.set_from_fen(START_POS);
        self.search_engine.reset_game();
    }

    pub fn set_position_from_fen(&mut self, fen_str: &str) {
        self.chess_position.set_from_fen(fen_str);
    }

    pub fn set_hash_size(&mut self, hash_size: usize) {
        self.search_engine.set_hash_size(hash_size);
    }

    #[inline]
    pub fn make_move(&mut self, chess_move: &ChessMove) {
        self.chess_position.make_move(chess_move);
    }

    pub fn perft(&mut self, depth: SearchDepth) {
        self.search_engine
            .perft(&mut self.chess_position, depth, &Instant::now());
    }

    pub fn search_best_move(
        &mut self,
        allowed_time_ms: MilliSeconds,
        extra_allowed_time_ms: MilliSeconds,
        force_stopped: Arc<AtomicBool>,
    ) -> ChessMove {
        self.search_engine.search_best_move(
            &mut self.chess_position,
            Duration::from_millis(allowed_time_ms),
            Duration::from_millis(extra_allowed_time_ms),
            None,
            force_stopped,
            true,
        )
    }

    #[inline(always)]
    pub fn get_position(&self) -> &ChessPosition<N> {
        &self.chess_position
    }

    pub fn get_debug_info(&self) -> String {
        self.chess_position.to_fen()
    }
}
