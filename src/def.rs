// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

use crate::types::{CastlingFlag, ChessFile, ChessPiece, ChessSquare, Player, Score};

pub const STACK_SIZE_BYTES: usize = 128 * 1024 * 1024;

pub const CHESS_SQUARE_COUNT: usize = 64;
pub const PIECE_TYPE_COUNT: usize = 13;
pub const PLAYER_COUNT: usize = 2;
pub const CHESS_FILE_COUNT: usize = 8;
pub const CHESS_RANK_COUNT: usize = 8;

pub const WHITE: Player = 0;
pub const BLACK: Player = 1;

pub const WHITE_INDEX: usize = 0;
pub const BLACK_INDEX: usize = 1;

pub const NO_PIECE: ChessPiece = 0;
pub const WP: ChessPiece = 1;
pub const WN: ChessPiece = 2;
pub const WB: ChessPiece = 3;
pub const WR: ChessPiece = 4;
pub const WQ: ChessPiece = 5;
pub const WK: ChessPiece = 6;
pub const BP: ChessPiece = 7;
pub const BN: ChessPiece = 8;
pub const BB: ChessPiece = 9;
pub const BR: ChessPiece = 10;
pub const BQ: ChessPiece = 11;
pub const BK: ChessPiece = 12;

pub const NO_SQUARE: ChessSquare = 99;
pub const A1: ChessSquare = 0;
pub const B1: ChessSquare = 1;
pub const C1: ChessSquare = 2;
pub const D1: ChessSquare = 3;
pub const E1: ChessSquare = 4;
pub const F1: ChessSquare = 5;
pub const G1: ChessSquare = 6;
pub const H1: ChessSquare = 7;
pub const A2: ChessSquare = 8;
pub const B2: ChessSquare = 9;
pub const C2: ChessSquare = 10;
pub const D2: ChessSquare = 11;
pub const E2: ChessSquare = 12;
pub const F2: ChessSquare = 13;
pub const G2: ChessSquare = 14;
pub const H2: ChessSquare = 15;
pub const A3: ChessSquare = 16;
pub const B3: ChessSquare = 17;
pub const C3: ChessSquare = 18;
pub const D3: ChessSquare = 19;
pub const E3: ChessSquare = 20;
pub const F3: ChessSquare = 21;
pub const G3: ChessSquare = 22;
pub const H3: ChessSquare = 23;
pub const A4: ChessSquare = 24;
pub const B4: ChessSquare = 25;
pub const C4: ChessSquare = 26;
pub const D4: ChessSquare = 27;
pub const E4: ChessSquare = 28;
pub const F4: ChessSquare = 29;
pub const G4: ChessSquare = 30;
pub const H4: ChessSquare = 31;
pub const A5: ChessSquare = 32;
pub const B5: ChessSquare = 33;
pub const C5: ChessSquare = 34;
pub const D5: ChessSquare = 35;
pub const E5: ChessSquare = 36;
pub const F5: ChessSquare = 37;
pub const G5: ChessSquare = 38;
pub const H5: ChessSquare = 39;
pub const A6: ChessSquare = 40;
pub const B6: ChessSquare = 41;
pub const C6: ChessSquare = 42;
pub const D6: ChessSquare = 43;
pub const E6: ChessSquare = 44;
pub const F6: ChessSquare = 45;
pub const G6: ChessSquare = 46;
pub const H6: ChessSquare = 47;
pub const A7: ChessSquare = 48;
pub const B7: ChessSquare = 49;
pub const C7: ChessSquare = 50;
pub const D7: ChessSquare = 51;
pub const E7: ChessSquare = 52;
pub const F7: ChessSquare = 53;
pub const G7: ChessSquare = 54;
pub const H7: ChessSquare = 55;
pub const A8: ChessSquare = 56;
pub const B8: ChessSquare = 57;
pub const C8: ChessSquare = 58;
pub const D8: ChessSquare = 59;
pub const E8: ChessSquare = 60;
pub const F8: ChessSquare = 61;
pub const G8: ChessSquare = 62;
pub const H8: ChessSquare = 63;

#[allow(dead_code)]
pub const FILE_A: ChessFile = 0;
#[allow(dead_code)]
pub const FILE_B: ChessFile = 1;
#[allow(dead_code)]
pub const FILE_C: ChessFile = 2;
#[allow(dead_code)]
pub const FILE_D: ChessFile = 3;
#[allow(dead_code)]
pub const FILE_E: ChessFile = 4;
#[allow(dead_code)]
pub const FILE_F: ChessFile = 5;
#[allow(dead_code)]
pub const FILE_G: ChessFile = 6;
#[allow(dead_code)]
pub const FILE_H: ChessFile = 7;

#[allow(dead_code)]
pub const RANK_1: ChessFile = 0;
#[allow(dead_code)]
pub const RANK_2: ChessFile = 1;
#[allow(dead_code)]
pub const RANK_3: ChessFile = 2;
#[allow(dead_code)]
pub const RANK_4: ChessFile = 3;
#[allow(dead_code)]
pub const RANK_5: ChessFile = 4;
#[allow(dead_code)]
pub const RANK_6: ChessFile = 5;
#[allow(dead_code)]
pub const RANK_7: ChessFile = 6;
#[allow(dead_code)]
pub const RANK_8: ChessFile = 7;

#[allow(dead_code)]
pub const UP_DELTA: ChessSquare = 8;
#[allow(dead_code)]
pub const UP_LEFT_DELTA: ChessSquare = 7;
#[allow(dead_code)]
pub const UP_RIGHT_DELTA: ChessSquare = 9;
#[allow(dead_code)]
pub const RIGHT_DELTA: ChessSquare = 1;

pub const CREATE_ENPASSANT_DELTA: ChessSquare = 16;

pub const CASTLING_FLAG_WHITE: CastlingFlag = 0b0011;
pub const CASTLING_FLAG_BLACK: CastlingFlag = 0b1100;
pub const CASTLING_FLAG_EMPTY: CastlingFlag = 0b0000;
pub const CASTLING_FLAG_FULL: CastlingFlag = 0b1111;

pub const CASTLING_FLAG_WHITE_KING_SIDE: CastlingFlag = 0b0001;
pub const CASTLING_FLAG_WHITE_QUEEN_SIDE: CastlingFlag = 0b0010;

pub const CASTLING_FLAG_BLACK_KING_SIDE: CastlingFlag = 0b0100;
pub const CASTLING_FLAG_BLACK_QUEEN_SIDE: CastlingFlag = 0b1000;

pub const MATE_SCORE: Score = 20_000;
pub const TERMINATE_SCORE: Score = 10_000;
pub const DRAW_SCORE: Score = 0;

pub const PIECE_VALS: [Score; PIECE_TYPE_COUNT] = [
    0, 100, 300, 300, 500, 1000, 10000, 100, 300, 300, 500, 1000, 10000,
];
