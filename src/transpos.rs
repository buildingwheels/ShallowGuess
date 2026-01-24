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

use crate::types::{
    BitBoard, ChessMove, ChessMoveCount, ChessMoveType, HashKey, MegaBytes, Score, SearchDepth,
    EMPTY_CHESS_MOVE,
};

pub const DEFAULT_HASH_SIZE_MB: MegaBytes = 1024;
pub const MIN_HASH_SIZE_MB: MegaBytes = 2;
pub const MAX_HASH_SIZE_MB: MegaBytes = 65536;
pub const HASH_ENTRY_PER_MB: usize = 32768;

const HASH_UTILIZATION_RATIO: usize = 1000;
const MAX_AGE_DIFFERENCE: ChessMoveCount = 255;

type CompressedChessMove = u32;

#[derive(Clone, Copy, PartialEq)]
pub enum HashFlag {
    LowBound,
    HighBound,
    Exact,
}

#[derive(Clone, Copy)]
pub struct TableEntry {
    pub key: HashKey,
    pub safety_check: BitBoard,
    pub score: Score,
    pub depth: SearchDepth,
    pub age: ChessMoveCount,
    pub flag: HashFlag,
    pub chess_move: ChessMove,
}

#[derive(Clone, Copy)]
pub struct InternalTableEntry {
    pub key: HashKey,
    pub safety_check: BitBoard,
    pub score: Score,
    pub depth: SearchDepth,
    pub age: ChessMoveCount,
    pub flag: HashFlag,
    pub chess_move: CompressedChessMove,
}

const EMPTY_INTERNAL_TABLE_ENTRY: InternalTableEntry = InternalTableEntry {
    key: 0,
    safety_check: 0,
    score: 0,
    depth: 0,
    age: 0,
    flag: HashFlag::Exact,
    chess_move: 0,
};

pub struct TranspositionTable {
    table_size: usize,
    internal_table: Vec<InternalTableEntry>,
    utilization_count: usize,
}

impl TranspositionTable {
    pub fn new(mb_size: MegaBytes) -> Self {
        let table_size = mb_size * HASH_ENTRY_PER_MB;

        TranspositionTable {
            table_size,
            internal_table: vec![EMPTY_INTERNAL_TABLE_ENTRY; table_size],
            utilization_count: 0,
        }
    }

    pub fn re_size(&mut self, new_size_mb: MegaBytes) {
        let table_size = new_size_mb * HASH_ENTRY_PER_MB;
        self.table_size = table_size;

        self.internal_table = vec![EMPTY_INTERNAL_TABLE_ENTRY; table_size];
        self.utilization_count = 0;
    }

    pub fn clear(&mut self) {
        self.internal_table = vec![EMPTY_INTERNAL_TABLE_ENTRY; self.table_size];
        self.utilization_count = 0;
    }

    pub fn get(&self, key: HashKey, safety_check: BitBoard) -> Option<TableEntry> {
        let index = key & (self.table_size - 1);
        let entry = &self.internal_table[index];

        if entry.key == key && entry.safety_check == safety_check {
            return Some(convert_to_external_entry(entry));
        }

        None
    }

    pub fn set(&mut self, new_entry: &TableEntry) {
        let index = new_entry.key & (self.table_size - 1);
        let new_internal_entry = convert_to_internal_entry(new_entry);

        let existing_entry = &mut self.internal_table[index];

        if existing_entry.key != 0 {
            let age_diff = new_internal_entry.age.wrapping_sub(existing_entry.age);
            if age_diff > MAX_AGE_DIFFERENCE {
                existing_entry.clone_from(&new_internal_entry);
                return;
            }
        }

        if existing_entry.key == 0 {
            self.utilization_count += 1;
            existing_entry.clone_from(&new_internal_entry);
        } else {
            if existing_entry.key == new_entry.key {
                existing_entry.clone_from(&new_internal_entry);
                return;
            }

            let should_replace = match (new_internal_entry.flag, existing_entry.flag) {
                (HashFlag::Exact, HashFlag::LowBound | HashFlag::HighBound) => true,
                (HashFlag::Exact, HashFlag::Exact) => {
                    new_internal_entry.depth > existing_entry.depth
                        || (new_internal_entry.depth == existing_entry.depth
                            && new_internal_entry.age < existing_entry.age)
                }
                _ => {
                    new_internal_entry.depth > existing_entry.depth
                        || (new_internal_entry.depth == existing_entry.depth
                            && new_internal_entry.age < existing_entry.age)
                }
            };

            if should_replace {
                existing_entry.clone_from(&new_internal_entry);
            }
        }
    }

    pub fn get_utilization_permil(&self) -> usize {
        self.utilization_count * HASH_UTILIZATION_RATIO / (self.table_size * 2)
    }
}

#[inline(always)]
fn convert_to_internal_entry(entry: &TableEntry) -> InternalTableEntry {
    InternalTableEntry {
        key: entry.key,
        safety_check: entry.safety_check,
        score: entry.score,
        depth: entry.depth,
        age: entry.age,
        flag: entry.flag,
        chess_move: compress_move(entry.chess_move),
    }
}

#[inline(always)]
fn convert_to_external_entry(entry: &InternalTableEntry) -> TableEntry {
    TableEntry {
        key: entry.key,
        safety_check: entry.safety_check,
        score: entry.score,
        depth: entry.depth,
        age: entry.age,
        flag: entry.flag,
        chess_move: decompress_move(entry.chess_move),
    }
}

#[inline(always)]
fn compress_move(chess_move: ChessMove) -> CompressedChessMove {
    if chess_move.is_empty() {
        return 0;
    }

    let from = chess_move.from_square as u32;
    let to = chess_move.to_square as u32;
    let promotion = chess_move.promotion_piece as u32;
    let move_type = match chess_move.move_type {
        ChessMoveType::Regular => 0,
        ChessMoveType::Castle => 1,
        ChessMoveType::EnPassant => 2,
        ChessMoveType::CreateEnPassant => 3,
        ChessMoveType::Promotion => 4,
    };

    (move_type << 24) | (promotion << 16) | (to << 8) | from
}

#[inline(always)]
fn decompress_move(compressed_move: CompressedChessMove) -> ChessMove {
    if compressed_move == 0 {
        return EMPTY_CHESS_MOVE;
    }

    let from_square = (compressed_move & 0xFF) as usize;
    let to_square = ((compressed_move >> 8) & 0xFF) as usize;
    let promotion_piece = ((compressed_move >> 16) & 0xFF) as u8;
    let move_type = match (compressed_move >> 24) & 0xF {
        0 => ChessMoveType::Regular,
        1 => ChessMoveType::Castle,
        2 => ChessMoveType::EnPassant,
        3 => ChessMoveType::CreateEnPassant,
        4 => ChessMoveType::Promotion,
        _ => ChessMoveType::Regular,
    };

    ChessMove {
        from_square,
        to_square,
        promotion_piece,
        move_type,
    }
}
