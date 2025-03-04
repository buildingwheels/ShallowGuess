use crate::types::{BitBoard, ChessMove, ChessMoveCount, HashKey, MegaBytes, Score, SearchDepth, EMPTY_CHESS_MOVE};

pub const DEFAULT_HASH_SIZE_MB: MegaBytes = 1024;
pub const MIN_HASH_SIZE_MB: MegaBytes = 2;
pub const MAX_HASH_SIZE_MB: MegaBytes = 65536;
pub const HASH_ENTRY_PER_MB: usize = 8192;

const HASH_UTILIZATION_RATIO: usize = 1000;

#[derive(Clone, Copy)]
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
    pub hash_age: ChessMoveCount,
    pub flag: HashFlag,
    pub chess_move: ChessMove,
}

const EMPTY_TABLE_ENTRY: TableEntry = TableEntry {
    key: 0,
    safety_check: 0,
    score: 0,
    depth: 0,
    hash_age: 0,
    flag: HashFlag::Exact,
    chess_move: EMPTY_CHESS_MOVE,
};

pub struct TranspositionTable {
    mod_base: HashKey,
    depth_preffered_table: Vec<TableEntry>,
    always_replacing_table: Vec<TableEntry>,
    utilization_count: usize,
}

impl TranspositionTable {
    pub fn new(mb_size: MegaBytes) -> Self {
        let size = mb_size * HASH_ENTRY_PER_MB;

        TranspositionTable {
            mod_base: size - 1,
            depth_preffered_table: vec![EMPTY_TABLE_ENTRY; size],
            always_replacing_table: vec![EMPTY_TABLE_ENTRY; size],
            utilization_count: 0,
        }
    }

    pub fn re_size(&mut self, new_size_mb: MegaBytes) {
        let new_size = new_size_mb * HASH_ENTRY_PER_MB;
        self.mod_base = new_size - 1;
        self.depth_preffered_table = vec![EMPTY_TABLE_ENTRY; new_size];
        self.always_replacing_table = vec![EMPTY_TABLE_ENTRY; new_size];
        self.utilization_count = 0;
    }

    pub fn clear(&mut self) {
        self.depth_preffered_table = vec![EMPTY_TABLE_ENTRY; self.mod_base + 1];
        self.always_replacing_table = vec![EMPTY_TABLE_ENTRY; self.mod_base + 1];
        self.utilization_count = 0;
    }

    pub fn get(&self, key: HashKey, safety_check: BitBoard) -> Option<&TableEntry> {
        let index = key & self.mod_base;
        let entry = &self.depth_preffered_table[index];

        if entry.key == key && entry.safety_check == safety_check {
            return Some(entry);
        }

        let entry = &self.always_replacing_table[index];

        if entry.key == key && entry.safety_check == safety_check {
            return Some(entry);
        }

        None
    }

    pub fn set(&mut self, new_entry: &TableEntry) {
        let index = new_entry.key & self.mod_base;
        let entry = &mut self.depth_preffered_table[index];

        if entry.key == 0 {
            self.utilization_count += 1;
        }

        if new_entry.depth + new_entry.hash_age >= entry.depth + entry.hash_age {
            entry.clone_from(&new_entry);
        } else {
            let entry = &mut self.always_replacing_table[index];
            entry.clone_from(&new_entry);
        }
    }

    pub fn get_utilization_permill(&self) -> usize {
        self.utilization_count * HASH_UTILIZATION_RATIO / self.mod_base
    }
}
