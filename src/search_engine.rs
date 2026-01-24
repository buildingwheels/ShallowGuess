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

use crate::chess_move_gen::{
    generate_captures_and_promotions, generate_quiet_moves, is_in_check, is_invalid_position,
    static_exchange_evaluation,
};
use crate::chess_position::ChessPosition;
use crate::def::{
    A1, BK, CHESS_SQUARE_COUNT, DRAW_SCORE, H8, MATE_SCORE, NO_PIECE, PIECE_TYPE_COUNT, PIECE_VALS,
    PLAYER_COUNT, TERMINATE_SCORE, WP,
};
use crate::fen::format_chess_move;
use crate::network::Network;
use crate::transpos::{HashFlag, TableEntry, TranspositionTable};
use crate::types::{
    BitBoard, ChessMove, ChessMoveCount, ChessMoveType, ChessPieceCount, HashKey, MilliSeconds,
    NodeCount, Player, Score, SearchDepth, SearchPly, SortableChessMove, EMPTY_CHESS_MOVE,
    MAX_PIECE_COUNT,
};
use crate::uci::print_info;
use crate::util::u16_sqrt;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct TwoBucketMoveTableEntry {
    pub primary: ChessMove,
    pub secondary: ChessMove,
    pub primary_depth: SearchDepth,
}

const EMPTY_TWO_BUCKET_MOVE_TABLE_ENTRY: TwoBucketMoveTableEntry = TwoBucketMoveTableEntry {
    primary: EMPTY_CHESS_MOVE,
    secondary: EMPTY_CHESS_MOVE,
    primary_depth: 0,
};

type HistoryTable = [[Score; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];
type ContinuationHistoryTable =
    [[[[Score; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT]; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];
type KillerTable = [[TwoBucketMoveTableEntry; MAX_PV_LENGTH]; PLAYER_COUNT];
type CounterMoveTable =
    [[[TwoBucketMoveTableEntry; CHESS_SQUARE_COUNT]; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];
type FollowupMoveTable =
    [[[TwoBucketMoveTableEntry; CHESS_SQUARE_COUNT]; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];

type NodeType = i8;

const NODE_TYPE_PV: NodeType = 0;
const NODE_TYPE_NON_PV: NodeType = 1;

const ASPIRATION_WINDOW_WIDTH_NORMAL: Score = 10;
const ASPIRATION_WINDOW_WIDTH_VOLATILE: Score = 50;

const VOLATILE_POS_THRESHOLD: Score = 100;

const HISTORY_DECAY_THRESHOLD: Score = 1024 * 1024;

const MAX_DEPTH: SearchDepth = 128;
const MAX_PV_LENGTH: usize = 128;
const MAX_SEARCH_TIME_MILLIS: MilliSeconds = 3600;

const SORT_PRIORITY_KILLER: Score = 2;
const SORT_PRIORITY_COUNTER_FOLLOWUP: Score = 1;
const SORT_PRIORITY_OTHER: Score = 0;

const HISTORY_WEIGHT_SHIFT_COUNTER: usize = 2;
const HISTORY_WEIGHT_SHIFT_FOLLOWUP: usize = 1;
const DISTANT_HISTORY_WEIGHT_SHIFT: usize = 2;

const NULL_MOVE_PRUNING_MIN_DEPTH: SearchDepth = 3;
const NULL_MOVE_VERIFICATION_DEPTH: SearchDepth = 6;
const NULL_MOVE_PRUNING_MARGIN: Score = 20;

const STATIC_PRUNING_MARGINS: [Score; MAX_PIECE_COUNT as usize + 1] = [
    500, 500, 350, 350, 300, 300, 300, 300, 300, 300, 250, 250, 200, 200, 150,
];

const ENDGAME_PIECE_COUNT: ChessPieceCount = 6;

pub struct SearchInfo {
    pub score: Score,
    pub depth: SearchDepth,
    pub searched_node_count: NodeCount,
    pub selected_depth: SearchPly,
    pub searched_time_ms: MilliSeconds,
    pub hash_utilization_permil: usize,
}

pub struct SearchEngine {
    transposition_table: TranspositionTable,

    killer_table: KillerTable,

    quiet_counter_move_table: CounterMoveTable,
    quiet_followup_move_table: FollowupMoveTable,

    non_quiet_counter_move_table: CounterMoveTable,
    non_quiet_followup_move_table: FollowupMoveTable,

    main_history_table: HistoryTable,
    continuation_history_table: ContinuationHistoryTable,

    searched_move_count: ChessMoveCount,
    searched_or_pruned_node_count: NodeCount,
    selected_depth: SearchPly,
    search_iteration_counter: ChessMoveCount,
    search_start_time: Instant,
    allowed_search_time: Duration,
    allowed_max_depth: SearchDepth,
    aborted: bool,
    force_stopped: Arc<AtomicBool>,
}

impl SearchEngine {
    pub fn new(transposition_table: TranspositionTable) -> Self {
        SearchEngine {
            transposition_table,

            killer_table: [[EMPTY_TWO_BUCKET_MOVE_TABLE_ENTRY; MAX_PV_LENGTH]; PLAYER_COUNT],

            quiet_counter_move_table: [[[EMPTY_TWO_BUCKET_MOVE_TABLE_ENTRY; CHESS_SQUARE_COUNT];
                CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT],
            quiet_followup_move_table: [[[EMPTY_TWO_BUCKET_MOVE_TABLE_ENTRY; CHESS_SQUARE_COUNT];
                CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT],

            non_quiet_counter_move_table: [[[EMPTY_TWO_BUCKET_MOVE_TABLE_ENTRY; CHESS_SQUARE_COUNT];
                CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT],
            non_quiet_followup_move_table: [[[EMPTY_TWO_BUCKET_MOVE_TABLE_ENTRY; CHESS_SQUARE_COUNT];
                CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT],

            main_history_table: [[0; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT],
            continuation_history_table: [[[[0; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];
                CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT],

            searched_move_count: 0,
            searched_or_pruned_node_count: 0,
            selected_depth: 0,

            search_start_time: Instant::now(),
            allowed_search_time: Duration::from_millis(0),
            allowed_max_depth: MAX_DEPTH,
            search_iteration_counter: 0,

            aborted: false,
            force_stopped: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn reset_game(&mut self) {
        self.searched_move_count = 0;
        self.force_stopped = Arc::new(AtomicBool::new(false));
        self.aborted = false;

        self.transposition_table.clear();

        self.quiet_counter_move_table = [[[EMPTY_TWO_BUCKET_MOVE_TABLE_ENTRY; CHESS_SQUARE_COUNT];
            CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];
        self.quiet_followup_move_table = [[[EMPTY_TWO_BUCKET_MOVE_TABLE_ENTRY; CHESS_SQUARE_COUNT];
            CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];
        self.non_quiet_counter_move_table = [[[EMPTY_TWO_BUCKET_MOVE_TABLE_ENTRY;
            CHESS_SQUARE_COUNT]; CHESS_SQUARE_COUNT];
            PIECE_TYPE_COUNT];
        self.non_quiet_followup_move_table = [[[EMPTY_TWO_BUCKET_MOVE_TABLE_ENTRY;
            CHESS_SQUARE_COUNT]; CHESS_SQUARE_COUNT];
            PIECE_TYPE_COUNT];
        self.killer_table = [[EMPTY_TWO_BUCKET_MOVE_TABLE_ENTRY; MAX_PV_LENGTH]; PLAYER_COUNT];
        self.main_history_table = [[0; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];
        self.continuation_history_table =
            [[[[0; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT]; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];
    }

    pub fn set_hash_size(&mut self, hash_size: usize) {
        self.transposition_table.re_size(hash_size);
    }

    pub fn search_best_move<N: Network>(
        &mut self,
        chess_position: &mut ChessPosition<N>,
        allowed_time: Duration,
        extra_allowed_time: Duration,
        allowed_max_depth: Option<SearchDepth>,
        force_stopped: Arc<AtomicBool>,
        show_output: bool,
    ) -> ChessMove {
        self.search_start_time = Instant::now();
        self.allowed_search_time = allowed_time;
        self.force_stopped = force_stopped;
        self.search_iteration_counter = self.search_iteration_counter.wrapping_add(1);

        self.searched_or_pruned_node_count = 0;
        self.selected_depth = 0;
        self.aborted = false;

        self.killer_table = [[EMPTY_TWO_BUCKET_MOVE_TABLE_ENTRY; MAX_PV_LENGTH]; PLAYER_COUNT];

        self.main_history_table = [[0; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];
        self.continuation_history_table =
            [[[[0; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT]; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];

        let soft_stop_time = allowed_time / 2;

        let in_check = is_in_check(chess_position);

        if in_check {
            let (has_forced_move, forced_move) = check_for_forced_move(chess_position, in_check);

            if has_forced_move {
                return forced_move;
            }
        }

        if let Some(max_depth) = allowed_max_depth {
            self.allowed_max_depth = max_depth;
        } else {
            self.allowed_max_depth = MAX_DEPTH;
        }

        let mut best_move = EMPTY_CHESS_MOVE;
        let mut depth = 1;
        let mut previous_score = 0;
        let mut searching_extended_window = false;
        let mut searching_full_window = false;
        let mut is_volatile_position = false;
        let mut used_extra_time = false;

        loop {
            let (alpha, beta) = if searching_full_window || depth == 1 {
                (-MATE_SCORE, MATE_SCORE)
            } else {
                if is_volatile_position || searching_extended_window {
                    (
                        previous_score - ASPIRATION_WINDOW_WIDTH_VOLATILE,
                        previous_score + ASPIRATION_WINDOW_WIDTH_VOLATILE,
                    )
                } else {
                    (
                        previous_score - ASPIRATION_WINDOW_WIDTH_NORMAL,
                        previous_score + ASPIRATION_WINDOW_WIDTH_NORMAL,
                    )
                }
            };

            let score = self.ab_search(
                chess_position,
                alpha,
                beta,
                NODE_TYPE_PV,
                in_check,
                depth,
                0,
            );

            if self.aborted {
                break;
            }

            if score <= alpha || score >= beta {
                if searching_extended_window {
                    searching_full_window = true;
                } else {
                    searching_extended_window = true;
                }

                continue;
            }

            let mut principal_variation = Vec::new();

            self.retrieve_principal_variation(
                chess_position,
                &mut principal_variation,
                depth as usize,
            );

            if principal_variation.is_empty() {
                self.transposition_table.clear();
                continue;
            }

            if show_output {
                self.print_search_info(score, depth, &principal_variation);
            }

            let current_best_move = principal_variation[0];

            if !used_extra_time {
                if self.searched_move_count == 0
                    || (current_best_move != best_move
                        && self.search_start_time.elapsed() > soft_stop_time)
                {
                    self.allowed_search_time += extra_allowed_time;
                    used_extra_time = true;
                }
            }

            best_move = current_best_move;

            if score > TERMINATE_SCORE || score < -TERMINATE_SCORE {
                break;
            }

            if !used_extra_time && self.search_start_time.elapsed() >= soft_stop_time {
                break;
            }

            if depth >= self.allowed_max_depth {
                break;
            }

            is_volatile_position = (score - previous_score).abs() > VOLATILE_POS_THRESHOLD;

            depth += 1;
            previous_score = score;
            searching_extended_window = false;
            searching_full_window = false;
        }

        self.searched_move_count += 1;

        best_move
    }

    pub fn search_to_depth<N: Network>(
        &mut self,
        chess_position: &mut ChessPosition<N>,
        depth: SearchDepth,
    ) -> Score {
        self.search_start_time = Instant::now();
        self.allowed_search_time = Duration::from_millis(MAX_SEARCH_TIME_MILLIS);

        self.ab_search(
            chess_position,
            -MATE_SCORE,
            MATE_SCORE,
            NODE_TYPE_PV,
            is_in_check(chess_position),
            depth,
            0,
        )
    }

    fn ab_search<N: Network>(
        &mut self,
        chess_position: &mut ChessPosition<N>,
        mut alpha: Score,
        mut beta: Score,
        node_type: NodeType,
        in_check: bool,
        depth: SearchDepth,
        ply: SearchPly,
    ) -> Score {
        if self.aborted {
            return alpha;
        }

        if self.force_stopped.load(Ordering::Relaxed)
            || self.search_start_time.elapsed() >= self.allowed_search_time
        {
            self.aborted = true;
            return alpha;
        }

        self.searched_or_pruned_node_count += 1;

        let safety_check = chess_position.white_all_bitboard | chess_position.black_all_bitboard;
        let age = self.search_iteration_counter;

        if ply > 0 {
            if chess_position.is_material_draw() {
                self.update_hash(&TableEntry {
                    key: chess_position.hash_key,
                    safety_check,
                    score: DRAW_SCORE,
                    depth: MAX_DEPTH,
                    age,
                    flag: HashFlag::Exact,
                    chess_move: EMPTY_CHESS_MOVE,
                });

                return DRAW_SCORE;
            }

            if chess_position.is_repetition_draw() {
                return DRAW_SCORE;
            }
        }

        if depth == 0 {
            if in_check {
                return self.ab_search(chess_position, alpha, beta, node_type, true, 1, ply);
            } else {
                return self.q_search(chess_position, alpha, beta, node_type, ply);
            }
        }

        let on_pv = node_type == NODE_TYPE_PV;

        let hash_move;

        if let Some(entry) = self.lookup_hash(chess_position.hash_key, safety_check) {
            hash_move = entry.chess_move;

            if !on_pv && entry.depth >= depth && entry.score != DRAW_SCORE {
                match entry.flag {
                    HashFlag::LowBound => {
                        if entry.score >= beta {
                            return entry.score;
                        }

                        if entry.score > alpha {
                            alpha = entry.score;
                        }
                    }
                    HashFlag::HighBound => {
                        if entry.score <= alpha {
                            return entry.score;
                        }

                        if entry.score < beta {
                            beta = entry.score;
                        }
                    }
                    HashFlag::Exact => {
                        return entry.score;
                    }
                }
            }
        } else {
            hash_move = EMPTY_CHESS_MOVE;
        }

        if !on_pv && !in_check && beta < TERMINATE_SCORE && depth == 1 {
            let static_pruning_score = chess_position.get_static_score()
                - STATIC_PRUNING_MARGINS[chess_position.get_piece_count() as usize];

            if static_pruning_score >= beta {
                self.update_hash(&TableEntry {
                    key: chess_position.hash_key,
                    safety_check,
                    score: static_pruning_score,
                    depth,
                    age,
                    flag: HashFlag::LowBound,
                    chess_move: EMPTY_CHESS_MOVE,
                });

                return static_pruning_score;
            }
        }

        let mut under_mate_threat = false;

        if !on_pv
            && !in_check
            && beta < TERMINATE_SCORE
            && depth >= NULL_MOVE_PRUNING_MIN_DEPTH
            && chess_position.get_static_score() - NULL_MOVE_PRUNING_MARGIN >= beta
        {
            let depth_reduction = u16_sqrt(depth << 1) - 1;

            let saved_enpassant_square = chess_position.make_null_move();

            let scout_score = -self.ab_search(
                chess_position,
                -beta,
                1 - beta,
                node_type,
                false,
                depth - depth_reduction - 1,
                ply + 1,
            );

            chess_position.unmake_null_move(saved_enpassant_square);

            if scout_score - NULL_MOVE_PRUNING_MARGIN >= beta && scout_score != DRAW_SCORE {
                if depth > NULL_MOVE_VERIFICATION_DEPTH
                    && chess_position.get_piece_count() >= ENDGAME_PIECE_COUNT
                {
                    self.update_hash(&TableEntry {
                        key: chess_position.hash_key,
                        safety_check,
                        score: scout_score,
                        depth,
                        age,
                        flag: HashFlag::LowBound,
                        chess_move: EMPTY_CHESS_MOVE,
                    });

                    return scout_score - NULL_MOVE_PRUNING_MARGIN;
                }

                let verification_score = self.ab_search(
                    chess_position,
                    beta - 1,
                    beta,
                    node_type,
                    false,
                    depth - depth_reduction,
                    ply,
                );

                if verification_score >= beta {
                    self.update_hash(&TableEntry {
                        key: chess_position.hash_key,
                        safety_check,
                        score: verification_score,
                        depth,
                        age,
                        flag: HashFlag::LowBound,
                        chess_move: EMPTY_CHESS_MOVE,
                    });

                    return verification_score;
                }
            } else if scout_score <= -TERMINATE_SCORE {
                under_mate_threat = true;
            }
        }

        let original_alpha = alpha;

        let mut valid_move_count = 0;

        let mut best_score = -MATE_SCORE + ply as Score;
        let mut best_move = EMPTY_CHESS_MOVE;

        if !hash_move.is_empty() {
            valid_move_count += 1;
            let saved_state = chess_position.make_move(&hash_move);

            let score = -self.ab_search(
                chess_position,
                -beta,
                -alpha,
                node_type,
                is_in_check(chess_position),
                depth - 1,
                ply + 1,
            );

            chess_position.unmake_move(&hash_move, saved_state);

            if self.aborted {
                return alpha;
            }

            let is_quiet_chess_move = is_quiet_chess_move(&hash_move, chess_position);

            if score >= beta {
                self.update_hash(&TableEntry {
                    key: chess_position.hash_key,
                    safety_check,
                    score,
                    depth,
                    age,
                    flag: HashFlag::LowBound,
                    chess_move: hash_move,
                });

                if is_quiet_chess_move {
                    self.update_killer_move(hash_move, chess_position.player, ply, depth);
                    self.update_quiet_counter_move(hash_move, chess_position, depth);
                    self.update_quiet_followup_move(hash_move, chess_position, depth);
                    self.update_history(&hash_move, chess_position, depth, None);
                } else {
                    self.update_non_quiet_counter_move(hash_move, chess_position, depth);
                    self.update_non_quiet_followup_move(hash_move, chess_position, depth);
                }

                return score;
            }

            if score > best_score {
                best_score = score;
                best_move = hash_move;

                if score > alpha {
                    alpha = score;

                    if is_quiet_chess_move {
                        self.update_quiet_followup_move(hash_move, chess_position, depth);
                    } else {
                        self.update_non_quiet_followup_move(hash_move, chess_position, depth);
                    }
                }
            }
        }

        let mut captures_and_promotions = self.sort_captures_and_promotions(
            chess_position,
            generate_captures_and_promotions(chess_position),
        );

        while let Some(sortable_chess_move) = captures_and_promotions.pop() {
            let chess_move = sortable_chess_move.chess_move;

            if chess_move == hash_move {
                continue;
            }

            let saved_state = chess_position.make_move(&chess_move);

            if is_invalid_position(chess_position) {
                chess_position.unmake_move(&chess_move, saved_state);
                continue;
            }

            valid_move_count += 1;

            let gives_check = is_in_check(chess_position);

            let mut score;

            if valid_move_count == 1 {
                score = -self.ab_search(
                    chess_position,
                    -beta,
                    -alpha,
                    node_type,
                    gives_check,
                    depth - 1,
                    ply + 1,
                );
            } else {
                score = -self.ab_search(
                    chess_position,
                    -alpha - 1,
                    -alpha,
                    NODE_TYPE_NON_PV,
                    gives_check,
                    depth - 1,
                    ply + 1,
                );

                if score > alpha && score < beta {
                    score = -self.ab_search(
                        chess_position,
                        -beta,
                        -alpha,
                        node_type,
                        gives_check,
                        depth - 1,
                        ply + 1,
                    );
                }
            }

            chess_position.unmake_move(&chess_move, saved_state);

            if self.aborted {
                return alpha;
            }

            if score >= beta {
                self.update_hash(&TableEntry {
                    key: chess_position.hash_key,
                    safety_check,
                    score,
                    depth,
                    age,
                    flag: HashFlag::LowBound,
                    chess_move,
                });

                self.update_non_quiet_counter_move(chess_move, chess_position, depth);
                self.update_non_quiet_followup_move(chess_move, chess_position, depth);

                return score;
            }

            if score > best_score {
                best_score = score;
                best_move = chess_move;

                if score > alpha {
                    alpha = score;

                    self.update_non_quiet_followup_move(chess_move, chess_position, depth);
                }
            }
        }

        let mut quiet_chess_moves = self.sort_quiet_moves(
            chess_position,
            generate_quiet_moves(chess_position, in_check),
            ply,
        );

        let mut valid_quiet_move_count = 0;
        let mut searched_quiet_chess_moves = Vec::new();

        while let Some(sortable_chess_move) = quiet_chess_moves.pop() {
            let chess_move = sortable_chess_move.chess_move;

            if chess_move == hash_move {
                continue;
            }

            let saved_state = chess_position.make_move(&chess_move);

            if is_invalid_position(chess_position) {
                chess_position.unmake_move(&chess_move, saved_state);
                continue;
            }

            valid_move_count += 1;
            valid_quiet_move_count += 1;

            let gives_check = is_in_check(chess_position);

            let mut score;

            if valid_move_count == 1 {
                score = -self.ab_search(
                    chess_position,
                    -beta,
                    -alpha,
                    node_type,
                    gives_check,
                    depth - 1,
                    ply + 1,
                );
            } else {
                let depth_reduction = if ply > 0 && !in_check && !under_mate_threat && depth > 1 {
                    if gives_check {
                        if sortable_chess_move.sort_score < 0 {
                            1
                        } else {
                            0
                        }
                    } else {
                        let mut reduction = u16_sqrt(depth + valid_quiet_move_count);

                        if on_pv {
                            reduction -= 1;
                        }

                        reduction.min(depth - 1)
                    }
                } else {
                    0
                };

                score = -self.ab_search(
                    chess_position,
                    -alpha - 1,
                    -alpha,
                    NODE_TYPE_NON_PV,
                    gives_check,
                    depth - depth_reduction - 1,
                    ply + 1,
                );

                if score > alpha && depth_reduction != 0 {
                    score = -self.ab_search(
                        chess_position,
                        -alpha - 1,
                        -alpha,
                        NODE_TYPE_NON_PV,
                        gives_check,
                        depth - 1,
                        ply + 1,
                    );
                }

                if score > alpha && score < beta {
                    score = -self.ab_search(
                        chess_position,
                        -beta,
                        -alpha,
                        node_type,
                        gives_check,
                        depth - 1,
                        ply + 1,
                    );
                }
            }

            chess_position.unmake_move(&chess_move, saved_state);

            if self.aborted {
                return alpha;
            }

            if score >= beta {
                self.update_hash(&TableEntry {
                    key: chess_position.hash_key,
                    safety_check,
                    score,
                    depth,
                    age,
                    flag: HashFlag::LowBound,
                    chess_move,
                });

                self.update_killer_move(chess_move, chess_position.player, ply, depth);
                self.update_quiet_counter_move(chess_move, chess_position, depth);
                self.update_quiet_followup_move(chess_move, chess_position, depth);
                self.update_history(
                    &chess_move,
                    chess_position,
                    depth,
                    Some(&searched_quiet_chess_moves),
                );

                return score;
            }

            if score > best_score {
                best_score = score;
                best_move = chess_move;

                if score > alpha {
                    alpha = score;

                    self.update_quiet_followup_move(chess_move, chess_position, depth);
                }
            }

            searched_quiet_chess_moves.push(chess_move);
        }

        if valid_move_count == 0 {
            let terminate_score = if in_check {
                -MATE_SCORE + ply as Score
            } else {
                DRAW_SCORE
            };

            self.update_hash(&TableEntry {
                key: chess_position.hash_key,
                safety_check,
                score: terminate_score,
                depth,
                age,
                flag: HashFlag::Exact,
                chess_move: EMPTY_CHESS_MOVE,
            });

            return terminate_score;
        }

        if alpha == original_alpha {
            self.update_hash(&TableEntry {
                key: chess_position.hash_key,
                safety_check,
                score: best_score,
                depth,
                age,
                flag: HashFlag::HighBound,
                chess_move: EMPTY_CHESS_MOVE,
            });
        } else {
            self.update_hash(&TableEntry {
                key: chess_position.hash_key,
                safety_check,
                score: best_score,
                depth,
                age,
                flag: HashFlag::Exact,
                chess_move: best_move,
            });
        }

        best_score
    }

    fn q_search<N: Network>(
        &mut self,
        chess_position: &mut ChessPosition<N>,
        mut alpha: Score,
        mut beta: Score,
        node_type: NodeType,
        ply: SearchPly,
    ) -> Score {
        if self.aborted {
            return alpha;
        }

        if self.search_start_time.elapsed() >= self.allowed_search_time {
            self.aborted = true;
            return alpha;
        }

        self.searched_or_pruned_node_count += 1;

        if ply > self.selected_depth {
            self.selected_depth = ply;
        }

        let safety_check = chess_position.white_all_bitboard | chess_position.black_all_bitboard;
        let age = self.search_iteration_counter;

        if chess_position.is_material_draw() {
            self.update_hash(&TableEntry {
                key: chess_position.hash_key,
                safety_check,
                score: DRAW_SCORE,
                depth: MAX_DEPTH,
                age,
                flag: HashFlag::Exact,
                chess_move: EMPTY_CHESS_MOVE,
            });

            return DRAW_SCORE;
        }

        let on_pv = node_type == NODE_TYPE_PV;

        let hash_move;

        if let Some(entry) = self.lookup_hash(chess_position.hash_key, safety_check) {
            if !is_quiet_chess_move(&entry.chess_move, chess_position) {
                hash_move = entry.chess_move;
            } else {
                hash_move = EMPTY_CHESS_MOVE;
            }

            if !on_pv && entry.score != DRAW_SCORE {
                match entry.flag {
                    HashFlag::LowBound => {
                        if entry.score >= beta {
                            return entry.score;
                        }

                        if entry.score > alpha {
                            alpha = entry.score;
                        }
                    }
                    HashFlag::HighBound => {
                        if entry.score <= alpha {
                            return entry.score;
                        }

                        if entry.score < beta {
                            beta = entry.score;
                        }
                    }
                    HashFlag::Exact => {
                        return entry.score;
                    }
                }
            }
        } else {
            hash_move = EMPTY_CHESS_MOVE;
        }

        let original_alpha = alpha;

        let mut best_score = chess_position.get_static_score();

        if best_score >= beta {
            self.update_hash(&TableEntry {
                key: chess_position.hash_key,
                safety_check,
                score: best_score,
                depth: 0,
                age,
                flag: HashFlag::LowBound,
                chess_move: EMPTY_CHESS_MOVE,
            });

            return best_score;
        }

        if best_score > alpha {
            alpha = best_score;
        }

        let mut best_move = EMPTY_CHESS_MOVE;
        let mut valid_move_count = 0;

        if !hash_move.is_empty() {
            valid_move_count += 1;

            let saved_state = chess_position.make_move(&hash_move);

            let score = if is_in_check(chess_position) {
                -self.ab_search(chess_position, -beta, -alpha, node_type, true, 1, ply + 1)
            } else {
                -self.q_search(chess_position, -beta, -alpha, node_type, ply + 1)
            };

            chess_position.unmake_move(&hash_move, saved_state);

            if self.aborted {
                return alpha;
            }

            if score >= beta {
                self.update_hash(&TableEntry {
                    key: chess_position.hash_key,
                    safety_check,
                    score,
                    depth: 0,
                    age,
                    flag: HashFlag::LowBound,
                    chess_move: hash_move,
                });

                return score;
            }

            if score > best_score {
                best_score = score;
                best_move = hash_move;

                if score > alpha {
                    alpha = score;
                }
            }
        }

        let mut captures_and_promotions = self.sort_captures_and_promotions(
            chess_position,
            generate_captures_and_promotions(chess_position),
        );

        while let Some(sortable_chess_move) = captures_and_promotions.pop() {
            let chess_move = sortable_chess_move.chess_move;

            if chess_move == hash_move {
                continue;
            }

            let saved_state = chess_position.make_move(&chess_move);

            if is_invalid_position(chess_position) {
                chess_position.unmake_move(&chess_move, saved_state);
                continue;
            }

            let gives_check = is_in_check(chess_position);

            if !on_pv && !gives_check && sortable_chess_move.sort_score < 0 {
                self.searched_or_pruned_node_count += 1;
                chess_position.unmake_move(&chess_move, saved_state);
                continue;
            }

            valid_move_count += 1;

            let score = if gives_check {
                if valid_move_count == 1 {
                    -self.ab_search(chess_position, -beta, -alpha, node_type, true, 1, ply + 1)
                } else {
                    let score = -self.ab_search(
                        chess_position,
                        -alpha - 1,
                        -alpha,
                        NODE_TYPE_NON_PV,
                        true,
                        1,
                        ply + 1,
                    );

                    if score > alpha && score < beta {
                        -self.ab_search(chess_position, -beta, -alpha, node_type, true, 1, ply + 1)
                    } else {
                        score
                    }
                }
            } else {
                if valid_move_count == 1 {
                    -self.q_search(chess_position, -beta, -alpha, node_type, ply + 1)
                } else {
                    let score = -self.q_search(
                        chess_position,
                        -alpha - 1,
                        -alpha,
                        NODE_TYPE_NON_PV,
                        ply + 1,
                    );

                    if score > alpha && score < beta {
                        -self.q_search(chess_position, -beta, -alpha, node_type, ply + 1)
                    } else {
                        score
                    }
                }
            };

            chess_position.unmake_move(&chess_move, saved_state);

            if self.aborted {
                return alpha;
            }

            if score >= beta {
                self.update_hash(&TableEntry {
                    key: chess_position.hash_key,
                    safety_check,
                    score,
                    depth: 0,
                    age,
                    flag: HashFlag::LowBound,
                    chess_move,
                });

                self.update_non_quiet_counter_move(chess_move, chess_position, 0);
                self.update_non_quiet_followup_move(chess_move, chess_position, 0);

                return score;
            }

            if score > best_score {
                best_score = score;
                best_move = chess_move;

                if score > alpha {
                    alpha = score;

                    self.update_non_quiet_followup_move(chess_move, chess_position, 0);
                }
            }
        }

        if alpha == original_alpha {
            self.update_hash(&TableEntry {
                key: chess_position.hash_key,
                safety_check,
                score: best_score,
                depth: 0,
                age,
                flag: HashFlag::HighBound,
                chess_move: EMPTY_CHESS_MOVE,
            });
        } else {
            self.update_hash(&TableEntry {
                key: chess_position.hash_key,
                safety_check,
                score: best_score,
                depth: 0,
                age,
                flag: HashFlag::Exact,
                chess_move: best_move,
            });
        }

        best_score
    }

    #[inline(always)]
    fn sort_captures_and_promotions<N: Network>(
        &self,
        chess_position: &mut ChessPosition<N>,
        chess_moves: Vec<ChessMove>,
    ) -> BinaryHeap<SortableChessMove> {
        let mut sorted_moves = BinaryHeap::with_capacity(chess_moves.len());

        let (primary_counter_move, secondary_counter_move) =
            self.get_non_quiet_counter_move(chess_position);

        let (primary_followup_move, secondary_followup_move) =
            self.get_non_quiet_followup_move(chess_position);

        for chess_move in chess_moves {
            let mvv_lva_score = get_mvv_lva_score(&chess_move, chess_position);

            if chess_move == primary_counter_move
                || chess_move == secondary_counter_move
                || chess_move == primary_followup_move
                || chess_move == secondary_followup_move
            {
                sorted_moves.push(SortableChessMove {
                    chess_move,
                    sort_score: mvv_lva_score,
                    priority: SORT_PRIORITY_COUNTER_FOLLOWUP,
                });

                continue;
            }

            if mvv_lva_score > 0 {
                sorted_moves.push(SortableChessMove {
                    chess_move,
                    sort_score: mvv_lva_score,
                    priority: SORT_PRIORITY_OTHER,
                });
            } else {
                sorted_moves.push(SortableChessMove {
                    chess_move,
                    sort_score: get_static_exchange_score(&chess_move, chess_position),
                    priority: SORT_PRIORITY_OTHER,
                });
            }
        }

        sorted_moves
    }

    #[inline(always)]
    fn sort_quiet_moves<N: Network>(
        &self,
        chess_position: &mut ChessPosition<N>,
        chess_moves: Vec<ChessMove>,
        ply: SearchPly,
    ) -> BinaryHeap<SortableChessMove> {
        let mut sorted_moves = BinaryHeap::with_capacity(chess_moves.len());

        let (primary_killer_current_ply, secondary_killer_current_ply) =
            self.get_killer_moves(chess_position.player, ply);

        let (primary_counter_move, secondary_counter_move) =
            self.get_quiet_counter_move(chess_position);

        let (primary_followup_move, secondary_followup_move) =
            self.get_quiet_followup_move(chess_position);

        for chess_move in chess_moves {
            let history_score = self.get_history(&chess_move, chess_position);

            if chess_move == primary_killer_current_ply
                || chess_move == secondary_killer_current_ply
            {
                sorted_moves.push(SortableChessMove {
                    chess_move,
                    sort_score: history_score,
                    priority: SORT_PRIORITY_KILLER,
                });

                continue;
            }

            if chess_move == primary_counter_move
                || chess_move == secondary_counter_move
                || chess_move == primary_followup_move
                || chess_move == secondary_followup_move
            {
                sorted_moves.push(SortableChessMove {
                    chess_move,
                    sort_score: history_score,
                    priority: SORT_PRIORITY_COUNTER_FOLLOWUP,
                });

                continue;
            }

            sorted_moves.push(SortableChessMove {
                chess_move,
                sort_score: history_score,
                priority: SORT_PRIORITY_OTHER,
            });
        }

        sorted_moves
    }
    pub fn perft<N: Network>(
        &mut self,
        chess_position: &mut ChessPosition<N>,
        depth: SearchDepth,
        start_time: &Instant,
    ) -> ChessMoveCount {
        if depth == 0 {
            return 1;
        }

        let mut chess_move_count = 0;

        let mut chess_moves = generate_captures_and_promotions(chess_position);
        chess_moves.append(&mut generate_quiet_moves(
            chess_position,
            is_in_check(chess_position),
        ));

        for chess_move in &chess_moves {
            let saved_state = chess_position.make_move(chess_move);

            if is_invalid_position(chess_position) {
                chess_position.unmake_move(chess_move, saved_state);
                continue;
            }

            let next_perft = self.perft_helper(chess_position, depth - 1, 0);

            println!("{}: {}", format_chess_move(chess_move), next_perft);

            chess_move_count += next_perft;
            chess_position.unmake_move(chess_move, saved_state);
        }

        println!("Total: {}", chess_move_count);
        println!(
            "Time: {}ms",
            Instant::now().duration_since(*start_time).as_millis()
        );

        chess_move_count
    }

    fn perft_helper<N: Network>(
        &mut self,
        chess_position: &mut ChessPosition<N>,
        depth: SearchDepth,
        ply: SearchPly,
    ) -> ChessMoveCount {
        if depth == 0 {
            return 1;
        }

        let safety_check = chess_position.white_all_bitboard | chess_position.black_all_bitboard;

        if let Some(entry) = self.lookup_hash(chess_position.hash_key, safety_check) {
            if entry.depth == depth {
                return entry.score as ChessMoveCount;
            }
        }

        let mut chess_move_count = 0;

        let mut chess_moves = generate_captures_and_promotions(chess_position);
        chess_moves.append(&mut generate_quiet_moves(
            chess_position,
            is_in_check(chess_position),
        ));

        for chess_move in &chess_moves {
            let saved_state = chess_position.make_move(chess_move);

            if is_invalid_position(chess_position) {
                chess_position.unmake_move(chess_move, saved_state);
                continue;
            }

            let next_perft = self.perft_helper(chess_position, depth - 1, ply + 1);

            chess_move_count += next_perft;
            chess_position.unmake_move(chess_move, saved_state);
        }

        self.update_hash(&TableEntry {
            key: chess_position.hash_key,
            safety_check,
            score: chess_move_count as Score,
            depth,
            age: self.search_iteration_counter,
            flag: HashFlag::Exact,
            chess_move: EMPTY_CHESS_MOVE,
        });

        chess_move_count
    }

    fn retrieve_principal_variation<N: Network>(
        &mut self,
        chess_position: &mut ChessPosition<N>,
        principal_variation: &mut Vec<ChessMove>,
        max_depth: usize,
    ) {
        if principal_variation.len() > max_depth {
            return;
        }

        if !principal_variation.is_empty() && chess_position.is_repetition_draw() {
            return;
        }

        if let Some(entry) = self.lookup_hash(
            chess_position.hash_key,
            chess_position.white_all_bitboard | chess_position.black_all_bitboard,
        ) {
            if entry.flag != HashFlag::Exact && !principal_variation.is_empty() {
                return;
            }

            let chess_move = entry.chess_move;

            if !chess_move.is_empty() {
                principal_variation.push(chess_move);

                let saved_state = chess_position.make_move(&chess_move);
                self.retrieve_principal_variation(chess_position, principal_variation, max_depth);
                chess_position.unmake_move(&chess_move, saved_state);
            }
        }
    }

    #[inline(always)]
    fn lookup_hash(&self, hash_key: HashKey, safety_check: BitBoard) -> Option<TableEntry> {
        self.transposition_table.get(hash_key, safety_check)
    }

    #[inline(always)]
    fn update_hash(&mut self, table_entry: &TableEntry) {
        self.transposition_table.set(table_entry);
    }

    #[inline(always)]
    fn get_killer_moves(&self, player: Player, ply: SearchPly) -> (ChessMove, ChessMove) {
        if ply < MAX_PV_LENGTH {
            let entry = self.killer_table[player as usize][ply];
            (entry.primary, entry.secondary)
        } else {
            (EMPTY_CHESS_MOVE, EMPTY_CHESS_MOVE)
        }
    }

    #[inline(always)]
    fn update_killer_move(
        &mut self,
        chess_move: ChessMove,
        player: Player,
        ply: SearchPly,
        depth: SearchDepth,
    ) {
        if ply < MAX_PV_LENGTH {
            let entry = &mut self.killer_table[player as usize][ply];

            if depth >= entry.primary_depth {
                entry.secondary = entry.primary;
                entry.primary = chess_move;
                entry.primary_depth = depth;
            } else {
                entry.secondary = chess_move;
            }
        }
    }

    #[inline(always)]
    fn get_quiet_counter_move<N: Network>(
        &self,
        chess_position: &ChessPosition<N>,
    ) -> (ChessMove, ChessMove) {
        if let Some(last_move) = chess_position.get_last_move_from_opponent() {
            let entry = &self.quiet_counter_move_table[last_move.chess_piece as usize]
                [last_move.from_square][last_move.to_square];

            (entry.primary, entry.secondary)
        } else {
            (EMPTY_CHESS_MOVE, EMPTY_CHESS_MOVE)
        }
    }

    #[inline(always)]
    fn update_quiet_counter_move<N: Network>(
        &mut self,
        chess_move: ChessMove,
        chess_position: &ChessPosition<N>,
        depth: SearchDepth,
    ) {
        if let Some(last_move) = chess_position.get_last_move_from_opponent() {
            let entry = &mut self.quiet_counter_move_table[last_move.chess_piece as usize]
                [last_move.from_square][last_move.to_square];

            if depth >= entry.primary_depth {
                entry.secondary = entry.primary;
                entry.primary = chess_move;
                entry.primary_depth = depth;
            } else {
                entry.secondary = chess_move;
            }
        }
    }

    #[inline(always)]
    fn get_quiet_followup_move<N: Network>(
        &self,
        chess_position: &ChessPosition<N>,
    ) -> (ChessMove, ChessMove) {
        if let Some(last_move) = chess_position.get_last_move_from_current_player() {
            let entry = &self.quiet_followup_move_table[last_move.chess_piece as usize]
                [last_move.from_square][last_move.to_square];

            (entry.primary, entry.secondary)
        } else {
            (EMPTY_CHESS_MOVE, EMPTY_CHESS_MOVE)
        }
    }

    #[inline(always)]
    fn update_quiet_followup_move<N: Network>(
        &mut self,
        chess_move: ChessMove,
        chess_position: &ChessPosition<N>,
        depth: SearchDepth,
    ) {
        if let Some(last_move) = chess_position.get_last_move_from_current_player() {
            let entry = &mut self.quiet_followup_move_table[last_move.chess_piece as usize]
                [last_move.from_square][last_move.to_square];

            if depth >= entry.primary_depth {
                entry.secondary = entry.primary;
                entry.primary = chess_move;
                entry.primary_depth = depth;
            } else {
                entry.secondary = chess_move;
            }
        }
    }

    #[inline(always)]
    fn get_non_quiet_counter_move<N: Network>(
        &self,
        chess_position: &ChessPosition<N>,
    ) -> (ChessMove, ChessMove) {
        if let Some(last_move) = chess_position.get_last_move_from_opponent() {
            let entry = &self.non_quiet_counter_move_table[last_move.chess_piece as usize]
                [last_move.from_square][last_move.to_square];

            (entry.primary, entry.secondary)
        } else {
            (EMPTY_CHESS_MOVE, EMPTY_CHESS_MOVE)
        }
    }

    #[inline(always)]
    fn update_non_quiet_counter_move<N: Network>(
        &mut self,
        chess_move: ChessMove,
        chess_position: &ChessPosition<N>,
        depth: SearchDepth,
    ) {
        if let Some(last_move) = chess_position.get_last_move_from_opponent() {
            let entry = &mut self.non_quiet_counter_move_table[last_move.chess_piece as usize]
                [last_move.from_square][last_move.to_square];

            if depth >= entry.primary_depth {
                entry.secondary = entry.primary;
                entry.primary = chess_move;
                entry.primary_depth = depth;
            } else {
                entry.secondary = chess_move;
            }
        }
    }

    #[inline(always)]
    fn get_non_quiet_followup_move<N: Network>(
        &self,
        chess_position: &ChessPosition<N>,
    ) -> (ChessMove, ChessMove) {
        if let Some(last_move) = chess_position.get_last_move_from_current_player() {
            let entry = &self.non_quiet_followup_move_table[last_move.chess_piece as usize]
                [last_move.from_square][last_move.to_square];

            (entry.primary, entry.secondary)
        } else {
            (EMPTY_CHESS_MOVE, EMPTY_CHESS_MOVE)
        }
    }

    #[inline(always)]
    fn update_non_quiet_followup_move<N: Network>(
        &mut self,
        chess_move: ChessMove,
        chess_position: &ChessPosition<N>,
        depth: SearchDepth,
    ) {
        if let Some(last_move) = chess_position.get_last_move_from_current_player() {
            let entry = &mut self.non_quiet_followup_move_table[last_move.chess_piece as usize]
                [last_move.from_square][last_move.to_square];

            if depth >= entry.primary_depth {
                entry.secondary = entry.primary;
                entry.primary = chess_move;
                entry.primary_depth = depth;
            } else {
                entry.secondary = chess_move;
            }
        }
    }

    #[inline(always)]
    fn get_history<N: Network>(
        &self,
        chess_move: &ChessMove,
        chess_position: &ChessPosition<N>,
    ) -> Score {
        let moving_piece = chess_position.board[chess_move.from_square] as usize;
        let history_score = self.main_history_table[moving_piece][chess_move.to_square];

        if let Some(last_move_from_opponent) = chess_position.get_last_move_from_opponent() {
            let counter_history_score = self.continuation_history_table
                [last_move_from_opponent.chess_piece as usize][last_move_from_opponent.to_square]
                [moving_piece][chess_move.to_square];

            if let Some(last_move_from_current_player) =
                chess_position.get_last_move_from_current_player()
            {
                let followup_history_score = self.continuation_history_table
                    [last_move_from_current_player.chess_piece as usize]
                    [last_move_from_current_player.to_square][moving_piece][chess_move.to_square];

                if let Some(second_last_move_from_opponent) =
                    chess_position.get_second_last_move_from_opponent()
                {
                    let distant_counter_history_score = self.continuation_history_table
                        [second_last_move_from_opponent.chess_piece as usize]
                        [second_last_move_from_opponent.to_square][moving_piece]
                        [chess_move.to_square];

                    if let Some(second_last_move_from_current_player) =
                        chess_position.get_second_last_move_from_current_player()
                    {
                        let distant_followup_history_score = self.continuation_history_table
                            [second_last_move_from_current_player.chess_piece as usize]
                            [second_last_move_from_current_player.to_square][moving_piece]
                            [chess_move.to_square];

                        history_score
                            + (counter_history_score << HISTORY_WEIGHT_SHIFT_COUNTER)
                            + (followup_history_score << HISTORY_WEIGHT_SHIFT_FOLLOWUP)
                            + (distant_counter_history_score >> DISTANT_HISTORY_WEIGHT_SHIFT)
                            + (distant_followup_history_score >> DISTANT_HISTORY_WEIGHT_SHIFT)
                    } else {
                        history_score
                            + (counter_history_score << HISTORY_WEIGHT_SHIFT_COUNTER)
                            + (followup_history_score << HISTORY_WEIGHT_SHIFT_FOLLOWUP)
                            + (distant_counter_history_score >> DISTANT_HISTORY_WEIGHT_SHIFT)
                    }
                } else {
                    history_score
                        + (counter_history_score << HISTORY_WEIGHT_SHIFT_COUNTER)
                        + (followup_history_score << HISTORY_WEIGHT_SHIFT_FOLLOWUP)
                }
            } else {
                history_score + (counter_history_score << HISTORY_WEIGHT_SHIFT_COUNTER)
            }
        } else {
            history_score
        }
    }

    #[inline(always)]
    fn update_history<N: Network>(
        &mut self,
        chess_move: &ChessMove,
        chess_position: &ChessPosition<N>,
        depth: SearchDepth,
        searched_quiet_moves: Option<&Vec<ChessMove>>,
    ) {
        let history_score_change = calculate_history_score_change(depth);
        let moving_piece = chess_position.board[chess_move.from_square] as usize;

        let main_entry = &mut self.main_history_table[moving_piece][chess_move.to_square];

        *main_entry += history_score_change;

        let mut should_decay_history = *main_entry > HISTORY_DECAY_THRESHOLD;

        if let Some(last_move_from_opponent) = chess_position.get_last_move_from_opponent() {
            let continuation_entry = &mut self.continuation_history_table
                [last_move_from_opponent.chess_piece as usize][last_move_from_opponent.to_square]
                [moving_piece][chess_move.to_square];

            *continuation_entry += history_score_change;
            should_decay_history =
                should_decay_history || *continuation_entry > HISTORY_DECAY_THRESHOLD;

            if let Some(last_move_from_current_player) =
                chess_position.get_last_move_from_current_player()
            {
                let continuation_entry = &mut self.continuation_history_table
                    [last_move_from_current_player.chess_piece as usize]
                    [last_move_from_current_player.to_square][moving_piece][chess_move.to_square];

                *continuation_entry += history_score_change;
                should_decay_history =
                    should_decay_history || *continuation_entry > HISTORY_DECAY_THRESHOLD;

                if let Some(second_last_move_from_opponent) =
                    chess_position.get_second_last_move_from_opponent()
                {
                    let continuation_entry = &mut self.continuation_history_table
                        [second_last_move_from_opponent.chess_piece as usize]
                        [second_last_move_from_opponent.to_square][moving_piece]
                        [chess_move.to_square];

                    *continuation_entry += history_score_change >> DISTANT_HISTORY_WEIGHT_SHIFT;
                    should_decay_history =
                        should_decay_history || *continuation_entry > HISTORY_DECAY_THRESHOLD;

                    if let Some(second_last_move_from_current_player) =
                        chess_position.get_second_last_move_from_current_player()
                    {
                        let continuation_entry = &mut self.continuation_history_table
                            [second_last_move_from_current_player.chess_piece as usize]
                            [second_last_move_from_current_player.to_square][moving_piece]
                            [chess_move.to_square];

                        *continuation_entry += history_score_change >> DISTANT_HISTORY_WEIGHT_SHIFT;
                        should_decay_history =
                            should_decay_history || *continuation_entry > HISTORY_DECAY_THRESHOLD;
                    }
                }
            }
        }

        if let Some(searched_quiet_moves) = searched_quiet_moves {
            let history_score_change = history_score_change >> 1;

            for chess_move in searched_quiet_moves {
                let moving_piece = chess_position.board[chess_move.from_square] as usize;

                let main_entry = &mut self.main_history_table[moving_piece][chess_move.to_square];

                *main_entry -= history_score_change;

                should_decay_history =
                    should_decay_history || *main_entry < -HISTORY_DECAY_THRESHOLD;

                if let Some(last_move_from_opponent) = chess_position.get_last_move_from_opponent()
                {
                    let counter_entry = &mut self.continuation_history_table
                        [last_move_from_opponent.chess_piece as usize]
                        [last_move_from_opponent.to_square][moving_piece][chess_move.to_square];

                    *counter_entry -= history_score_change;
                    should_decay_history =
                        should_decay_history || *counter_entry < -HISTORY_DECAY_THRESHOLD;

                    if let Some(last_move_from_current_player) =
                        chess_position.get_last_move_from_current_player()
                    {
                        let followup_entry = &mut self.continuation_history_table
                            [last_move_from_current_player.chess_piece as usize]
                            [last_move_from_current_player.to_square][moving_piece]
                            [chess_move.to_square];

                        *followup_entry -= history_score_change;
                        should_decay_history =
                            should_decay_history || *followup_entry < -HISTORY_DECAY_THRESHOLD;

                        if let Some(second_last_move_from_opponent) =
                            chess_position.get_second_last_move_from_opponent()
                        {
                            let continuation_entry = &mut self.continuation_history_table
                                [second_last_move_from_opponent.chess_piece as usize]
                                [second_last_move_from_opponent.to_square][moving_piece]
                                [chess_move.to_square];

                            *continuation_entry -=
                                history_score_change >> DISTANT_HISTORY_WEIGHT_SHIFT;
                            should_decay_history = should_decay_history
                                || *continuation_entry < -HISTORY_DECAY_THRESHOLD;

                            if let Some(second_last_move_from_current_player) =
                                chess_position.get_second_last_move_from_current_player()
                            {
                                let continuation_entry = &mut self.continuation_history_table
                                    [second_last_move_from_current_player.chess_piece as usize]
                                    [second_last_move_from_current_player.to_square][moving_piece]
                                    [chess_move.to_square];

                                *continuation_entry -=
                                    history_score_change >> DISTANT_HISTORY_WEIGHT_SHIFT;
                                should_decay_history = should_decay_history
                                    || *continuation_entry < -HISTORY_DECAY_THRESHOLD;
                            }
                        }
                    }
                }
            }
        }

        if should_decay_history {
            self.decay_history();
        }
    }

    #[inline(always)]
    fn decay_history(&mut self) {
        for moving_piece in WP as usize..=BK as usize {
            for to_square in A1..=H8 {
                self.main_history_table[moving_piece][to_square] >>= 1;
            }
        }

        for prev_piece in WP as usize..=BK as usize {
            for prev_to_square in A1..=H8 {
                for current_piece in WP as usize..=BK as usize {
                    for current_to_square in A1..=H8 {
                        self.continuation_history_table[prev_piece][prev_to_square]
                            [current_piece][current_to_square] >>= 1;
                    }
                }
            }
        }
    }

    fn print_search_info(
        &self,
        score: Score,
        depth: SearchDepth,
        principal_variation: &Vec<ChessMove>,
    ) {
        print_info(
            SearchInfo {
                score,
                depth,
                searched_node_count: self.searched_or_pruned_node_count,
                selected_depth: self.selected_depth,
                searched_time_ms: self.search_start_time.elapsed().as_millis() as MilliSeconds,
                hash_utilization_permil: self.transposition_table.get_utilization_permil(),
            },
            principal_variation,
        );
    }
}

fn check_for_forced_move<N: Network>(
    chess_position: &mut ChessPosition<N>,
    in_check: bool,
) -> (bool, ChessMove) {
    let mut legal_chess_moves = Vec::new();

    let mut chess_moves = generate_captures_and_promotions(chess_position);
    chess_moves.append(&mut generate_quiet_moves(chess_position, in_check));

    for chess_move in &chess_moves {
        let saved_state = chess_position.make_move(chess_move);

        if !is_invalid_position(chess_position) {
            legal_chess_moves.push(*chess_move);
        }

        chess_position.unmake_move(chess_move, saved_state);
    }

    if legal_chess_moves.len() == 1 {
        return (true, *legal_chess_moves.first().unwrap());
    }

    (false, EMPTY_CHESS_MOVE)
}

#[inline(always)]
fn get_static_exchange_score<N: Network>(
    chess_move: &ChessMove,
    chess_position: &mut ChessPosition<N>,
) -> Score {
    let initial_gain = if chess_move.move_type == ChessMoveType::EnPassant {
        PIECE_VALS[WP as usize]
    } else {
        if chess_move.move_type == ChessMoveType::Promotion {
            PIECE_VALS[chess_position.board[chess_move.to_square] as usize]
                + PIECE_VALS[chess_move.promotion_piece as usize]
        } else {
            PIECE_VALS[chess_position.board[chess_move.to_square] as usize]
        }
    };

    let saved_state = chess_position.make_move(chess_move);
    let see_score = static_exchange_evaluation(chess_move.to_square, chess_position);
    chess_position.unmake_move(chess_move, saved_state);

    initial_gain - see_score
}

#[inline(always)]
fn get_mvv_lva_score(
    chess_move: &ChessMove,
    chess_position: &ChessPosition<impl Network>,
) -> Score {
    if chess_move.move_type == ChessMoveType::EnPassant {
        0
    } else {
        if chess_move.move_type == ChessMoveType::Promotion {
            PIECE_VALS[chess_position.board[chess_move.to_square] as usize]
                + PIECE_VALS[chess_move.promotion_piece as usize]
                - PIECE_VALS[chess_position.board[chess_move.from_square] as usize]
        } else {
            PIECE_VALS[chess_position.board[chess_move.to_square] as usize]
                - PIECE_VALS[chess_position.board[chess_move.from_square] as usize]
        }
    }
}

#[inline(always)]
fn is_quiet_chess_move<N: Network>(
    chess_move: &ChessMove,
    chess_position: &ChessPosition<N>,
) -> bool {
    chess_move.move_type != ChessMoveType::Promotion
        && chess_move.move_type != ChessMoveType::EnPassant
        && chess_position.board[chess_move.to_square] == NO_PIECE
}

#[inline(always)]
fn calculate_history_score_change(depth: SearchDepth) -> Score {
    (depth as Score) * (depth as Score)
}
