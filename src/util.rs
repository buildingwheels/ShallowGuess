// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

use crate::def::{
    A1, A2, A3, A4, A5, A6, A7, A8, B1, B2, B3, B4, B5, B6, B7, B8, BB, BK, BN, BP, BQ, BR, C1, C2,
    C3, C4, C5, C6, C7, C8, CHESS_SQUARE_COUNT, D1, D2, D3, D4, D5, D6, D7, D8, E1, E2, E3, E4, E5,
    E6, E7, E8, F1, F2, F3, F4, F5, F6, F7, F8, FILE_A, FILE_H, G1, G2, G3, G4, G5, G6, G7, G8, H1,
    H2, H3, H4, H5, H6, H7, H8, NO_PIECE, PIECE_TYPE_COUNT, WB, WK, WN, WP, WQ, WR,
};
use crate::network::NetworkIntValue;
use crate::types::{BitBoard, ChessFile, ChessPiece, ChessRank, ChessSquare};
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::path::Path;

pub type NetworkInputs = Vec<NetworkIntValue>;

#[inline(always)]
pub const fn get_file(chess_square: ChessSquare) -> ChessFile {
    chess_square & 7
}

#[inline(always)]
pub const fn get_rank(chess_square: ChessSquare) -> ChessRank {
    chess_square >> 3
}

#[inline(always)]
pub const fn get_lowest_occupied_chess_square(bitboard: BitBoard) -> ChessSquare {
    bitboard.trailing_zeros() as ChessSquare
}

#[inline(always)]
pub const fn get_highest_occupied_chess_square(bitboard: BitBoard) -> ChessSquare {
    H8 - bitboard.leading_zeros() as ChessSquare
}

#[inline(always)]
#[allow(dead_code)]
pub const fn i_count_set_bits(bit_board: BitBoard) -> isize {
    bit_board.count_ones() as isize
}

#[inline(always)]
#[allow(dead_code)]
pub const fn u_count_set_bits(bit_board: BitBoard) -> usize {
    bit_board.count_ones() as usize
}

#[inline(always)]
pub fn char_to_digit(c: char) -> usize {
    c.to_digit(10).unwrap() as usize
}

#[inline(always)]
pub fn digit_to_char(index: usize) -> char {
    (index as u8 + b'0') as char
}

const U16_SQRT_TABLE_SIZE: usize = 512;
const U16_SQRT_MAX: u16 = 256;
const U16_SQRT_TABLE: [u16; U16_SQRT_TABLE_SIZE] = precompute_u16_sqrt_table();

const fn precompute_u16_sqrt_table() -> [u16; U16_SQRT_TABLE_SIZE] {
    let mut table = [0u16; U16_SQRT_TABLE_SIZE];
    let mut index = 0;
    while index < U16_SQRT_TABLE_SIZE {
        let mut value = 0u16;

        while (value as u32 + 1) * (value as u32 + 1) <= index as u32 {
            value += 1;
        }

        table[index] = value;
        index += 1;
    }

    table
}

#[inline(always)]
pub fn u16_sqrt(value: u16) -> u16 {
    if value <= 1 {
        return value;
    }

    let index = value as usize;

    if index < U16_SQRT_TABLE_SIZE {
        U16_SQRT_TABLE[index]
    } else {
        u16_sqrt_fallback(value)
    }
}

#[cold]
#[inline(never)]
fn u16_sqrt_fallback(value: u16) -> u16 {
    let mut low = 0;
    let mut high = value.min(U16_SQRT_MAX);
    let mut result = 0;

    while low <= high {
        let mid = ((low as u32 + high as u32) >> 1) as u16;
        let mid_squared = mid as u32 * mid as u32;

        if mid_squared == value as u32 {
            return mid;
        } else if mid_squared < value as u32 {
            low = mid + 1;
            result = mid;
        } else {
            high = mid - 1;
        }
    }

    result
}

#[allow(dead_code)]
pub fn print_bitboard(bitboard: BitBoard) {
    let mut chess_square = A8;

    loop {
        if is_bit_set(bitboard, chess_square) {
            print!("1");
        } else {
            print!("0");
        }

        let chess_file = get_file(chess_square);

        if chess_file == FILE_H {
            if chess_square == H1 {
                break;
            }

            println!();

            chess_square -= 15;
        } else {
            chess_square += 1;
        }
    }

    println!();
}

#[allow(dead_code)]
pub fn print_bitboard_inline(bitboard: BitBoard) {
    let mut chess_square = H8;

    loop {
        if is_bit_set(bitboard, chess_square) {
            print!("1");
        } else {
            print!("0");
        }

        if chess_square == 0 {
            break;
        }

        let chess_file = get_file(chess_square);

        if chess_file == FILE_A {
            print!("_");
        }

        chess_square -= 1;
    }
}

#[allow(dead_code)]
fn is_bit_set(bitboard: BitBoard, index: usize) -> bool {
    bitboard & 1 << index != 0
}

#[inline(always)]
pub fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

#[macro_export]
macro_rules! process_occupied_indices {
    ($bitboard:expr, $action:expr) => {
        let mut next_set_index = 0;
        let mut bitboard = $bitboard;

        while bitboard != 0 {
            let trailing_zeros = bitboard.trailing_zeros();
            next_set_index += trailing_zeros;

            $action(next_set_index as ChessSquare);

            bitboard >>= trailing_zeros;
            bitboard -= 1;
        }
    };
}

#[macro_export]
macro_rules! process_occupied_indices_breakable {
    ($bitboard:expr, $action:expr) => {{
        let mut result = None;
        let mut bitboard = $bitboard;
        let mut next_set_index = 0;

        while bitboard != 0 {
            let trailing_zeros = bitboard.trailing_zeros();
            next_set_index += trailing_zeros;

            if let Some(res) = $action(next_set_index as ChessSquare) {
                result = Some(res);
                break;
            }

            bitboard >>= trailing_zeros;
            bitboard -= 1;
        }

        result
    }};
}

pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub const FLIPPED_CHESS_SQUARES: [ChessSquare; CHESS_SQUARE_COUNT] = [
    A8, B8, C8, D8, E8, F8, G8, H8, A7, B7, C7, D7, E7, F7, G7, H7, A6, B6, C6, D6, E6, F6, G6, H6,
    A5, B5, C5, D5, E5, F5, G5, H5, A4, B4, C4, D4, E4, F4, G4, H4, A3, B3, C3, D3, E3, F3, G3, H3,
    A2, B2, C2, D2, E2, F2, G2, H2, A1, B1, C1, D1, E1, F1, G1, H1,
];

pub const MIRRORED_CHESS_PIECES: [ChessPiece; PIECE_TYPE_COUNT] =
    [NO_PIECE, BP, BN, BB, BR, BQ, BK, WP, WN, WB, WR, WQ, WK];
