// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

use crate::def::{
    BB, BK, BN, BP, BQ, BR, CHESS_FILE_COUNT, NO_PIECE, WB, WHITE, WK, WN, WP, WQ, WR,
};
use crate::types::{
    ChessMove, ChessPiece, ChessPieceToChessPieceCharMap, ChessSquare, ChessSquareIndexToStrMap,
    Player,
};

pub mod fen_str_constants {
    pub const SPLITTER: &str = " ";

    pub const PLAYER_WHITE: &str = "w";
    pub const PLAYER_BLACK: &str = "b";

    pub const CASTLING_FLAG_WHITE_KING_SIDE: &str = "K";
    pub const CASTLING_FLAG_WHITE_QUEEN_SIDE: &str = "Q";
    pub const CASTLING_FLAG_BLACK_KING_SIDE: &str = "k";
    pub const CASTLING_FLAG_BLACK_QUEEN_SIDE: &str = "q";

    pub const NA_STR: &str = "-";

    pub const RANK_BREAK: char = '/';

    pub const START_POS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
}

pub const CHESS_SQUARE_INDEX_TO_STR_MAP: ChessSquareIndexToStrMap = [
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5", "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
];

pub const CHESS_PIECE_TO_CHESS_PIECE_CHAR_MAP: ChessPieceToChessPieceCharMap = [
    '-', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k',
];

pub const CHESS_PIECE_TO_PROMOTION_PIECE_CHAR_MAP: ChessPieceToChessPieceCharMap = [
    '-', '-', 'n', 'b', 'r', 'q', '-', '-', 'n', 'b', 'r', 'q', '-',
];

#[inline(always)]
pub fn get_chess_square_from_chars(file_char: char, rank_char: char) -> ChessSquare {
    let file = file_char as usize - 'a' as usize;
    let rank = rank_char.to_digit(10).unwrap() as usize - 1;

    rank * CHESS_FILE_COUNT + file
}

#[inline(always)]
pub fn get_chess_square_from_str(chess_square_str: &str) -> ChessSquare {
    get_chess_square_from_chars(
        chess_square_str.chars().nth(0).unwrap(),
        chess_square_str.chars().nth(1).unwrap(),
    )
}

#[inline(always)]
pub fn get_chess_piece_from_char(chess_piece_char: char) -> ChessPiece {
    match chess_piece_char {
        'K' => WK,
        'Q' => WQ,
        'R' => WR,
        'B' => WB,
        'N' => WN,
        'P' => WP,
        'k' => BK,
        'q' => BQ,
        'r' => BR,
        'b' => BB,
        'n' => BN,
        'p' => BP,
        _ => NO_PIECE,
    }
}

#[inline(always)]
pub fn get_promotion_chess_piece_from_char(
    promotion_chess_piece_char: char,
    player: Player,
) -> ChessPiece {
    if player == WHITE {
        match promotion_chess_piece_char {
            'q' => WQ,
            'r' => WR,
            'b' => WB,
            'n' => WN,
            _ => NO_PIECE,
        }
    } else {
        match promotion_chess_piece_char {
            'q' => BQ,
            'r' => BR,
            'b' => BB,
            'n' => BN,
            _ => NO_PIECE,
        }
    }
}

#[inline(always)]
pub fn format_chess_move(chess_move: &ChessMove) -> String {
    let mut output = String::new();

    output.push_str(CHESS_SQUARE_INDEX_TO_STR_MAP[chess_move.from_square]);
    output.push_str(CHESS_SQUARE_INDEX_TO_STR_MAP[chess_move.to_square]);

    if chess_move.promotion_piece != NO_PIECE {
        output.push(CHESS_PIECE_TO_PROMOTION_PIECE_CHAR_MAP[chess_move.promotion_piece as usize]);
    }

    output
}
