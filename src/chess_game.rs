use crate::chess_position::ChessPosition;
use crate::fen::fen_str_constants::START_POS;
use crate::search_engine::SearchEngine;
use crate::types::{ChessMove, MilliSeconds, SearchDepth, EMPTY_CHESS_MOVE};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct ChessGame {
    chess_position: ChessPosition,
    search_engine: SearchEngine,
    expected_responding_move: ChessMove,
    search_count: usize,
}

impl ChessGame {
    pub fn new(chess_position: ChessPosition, search_engine: SearchEngine) -> Self {
        ChessGame {
            chess_position,
            search_engine,
            expected_responding_move: EMPTY_CHESS_MOVE,
            search_count: 0,
        }
    }

    pub fn reset_game(&mut self) {
        self.chess_position.set_from_fen(START_POS);
        self.search_engine.reset_game();
        self.search_count = 0;
    }

    pub fn set_position_from_fen(&mut self, fen_str: &str) {
        self.chess_position.set_from_fen(fen_str);
    }

    pub fn set_hash_size(&mut self, hash_size: usize) {
        if hash_size & (hash_size - 1) != 0 {
            println!("Size {} not supported, needs to be power of 2", hash_size);
        } else {
            self.search_engine.set_hash_size(hash_size);
        }
    }

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
        force_stopped: Arc<AtomicBool>,
    ) -> ChessMove {
        self.search_count += 1;

        let (best_move, expected_responding_move) = self.search_engine.search_best_move(
            &mut self.chess_position,
            Duration::from_millis(allowed_time_ms),
            force_stopped,
            true,
        );

        self.expected_responding_move = expected_responding_move;

        best_move
    }

    pub fn get_position(&self) -> &ChessPosition {
        &self.chess_position
    }

    pub fn get_debug_info(&self) -> String {
        format!("{}", self.chess_position.to_fen())
    }

    pub fn require_extra_search_time(&self) -> bool {
        self.search_count == 0
            || (!self.expected_responding_move.is_empty()
                && *self.chess_position.get_last_move() != self.expected_responding_move)
    }
}
