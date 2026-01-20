// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

use std::cmp::Ordering;

use crate::def::{CHESS_SQUARE_COUNT, NO_PIECE, NO_SQUARE, PIECE_TYPE_COUNT};

pub type ChessSquare = usize;
pub type ChessFile = usize;
pub type ChessRank = usize;
pub type ChessPiece = u8;
pub type BitBoard = u64;
pub type Player = u8;
pub type CastlingFlag = u8;
pub type ChessMoveCount = u16;
pub type NodeCount = usize;
pub type SearchDepth = u16;
pub type SearchPly = usize;
pub type MilliSeconds = u64;
pub type MegaBytes = usize;
pub type Score = i32;
pub type HashKey = usize;
pub type ChessPieceCount = u32;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum ChessMoveType {
    Regular,
    Castle,
    EnPassant,
    CreateEnPassant,
    Promotion,
}

#[derive(Clone, Copy, Debug)]
pub struct ChessMove {
    pub from_square: ChessSquare,
    pub to_square: ChessSquare,
    pub promotion_piece: ChessPiece,
    pub move_type: ChessMoveType,
}

impl ChessMove {
    pub fn is_empty(&self) -> bool {
        self.from_square == self.to_square
    }
}

impl PartialEq<ChessMove> for ChessMove {
    fn eq(&self, other: &ChessMove) -> bool {
        self.from_square == other.from_square && self.to_square == other.to_square
    }
}

pub const EMPTY_CHESS_MOVE: ChessMove = ChessMove {
    from_square: 0,
    to_square: 0,
    promotion_piece: 0,
    move_type: ChessMoveType::Regular,
};

pub struct SortableChessMove {
    pub chess_move: ChessMove,
    pub priority: Score,
    pub sort_score: Score,
}

impl PartialEq for SortableChessMove {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sort_score == other.sort_score
    }
}

impl Eq for SortableChessMove {}

impl PartialOrd for SortableChessMove {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SortableChessMove {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => self.sort_score.cmp(&other.sort_score),

            other_ordering => other_ordering,
        }
    }
}

pub type ChessSquareIndexToStrMap = [&'static str; CHESS_SQUARE_COUNT];
pub type ChessPieceToChessPieceCharMap = [char; PIECE_TYPE_COUNT];

pub struct HistoryMove {
    pub chess_piece: ChessPiece,
    pub from_square: ChessSquare,
    pub to_square: ChessSquare,
}

pub const EMPTY_HISTORY_MOVE: HistoryMove = HistoryMove {
    chess_piece: NO_PIECE,
    from_square: NO_SQUARE,
    to_square: NO_SQUARE,
};
