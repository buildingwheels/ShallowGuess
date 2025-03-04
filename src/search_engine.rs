use crate::chess_move_gen::{
    generate_captures_and_promotions, generate_quiet_moves,
    get_least_valued_attackers_to_chess_square, is_in_check, is_invalid_position,
};
use crate::chess_position::ChessPosition;
use crate::def::{
    A1, BK, BLACK, CHESS_SQUARE_COUNT, H8, MATE_SCORE, PIECE_TYPE_COUNT, PLAYER_COUNT,
    TERMINATE_SCORE, WP,
};
use crate::fen::format_chess_move;
use crate::transpos::{HashFlag, TableEntry, TranspositionTable};
use crate::types::{
    BitBoard, ChessMove, ChessMoveCount, HashKey, MilliSeconds, NodeCount, Score, SearchDepth,
    SearchPly, SortableChessMove, EMPTY_CHESS_MOVE,
};
use crate::uci::print_info;
use crate::util::u16_sqrt;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

type HistoryTable = [[Score; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];
type KillerTable = [[[ChessMove; KILLER_COUNT]; MAX_PV_LENGTH]; PLAYER_COUNT];
type CounterMoveTable = [[[ChessMove; CHESS_SQUARE_COUNT]; CHESS_SQUARE_COUNT]; PLAYER_COUNT];

const WINDOW_SIZE: Score = 50;

const HISTORY_SCORE_SHIFT_FACTOR_CUTOFF: usize = 7;
const HISTORY_SCORE_SHIFT_FACTOR_RAISE: usize = 1;
const HISTORY_SCORE_SHIFT_FACTOR_DECAY: usize = 1;
const HISTORY_DECAY_INTERVAL: NodeCount = 1023;

const KILLER_COUNT: usize = 2;
const KILLER_INDEX_CUTOFF: usize = 0;
const KILLER_INDEX_RAISE: usize = 1;

const NULL_MOVE_PRUNING_MIN_DEPTH: SearchDepth = 5;
const NULL_MOVE_PRUNING_DEPTH_REDUCTION: SearchDepth = 2;

const FUTILITY_PRUNING_MAX_DEPTH: SearchDepth = 7;
const FUTILITY_PRUNING_MARGINS: [Score; 7] = [0, 200, 400, 600, 800, 1000, 1200];

const IID_MIN_DEPTH: SearchDepth = 7;
const IID_SEARCH_DEPTH: SearchDepth = 2;

const MAX_PV_LENGTH: usize = 128;

const SORTING_PIECE_VALS: [Score; PIECE_TYPE_COUNT] = [0, 1, 3, 3, 5, 10, 100, 1, 3, 3, 5, 10, 100];

pub struct SearchInfo {
    pub score: Score,
    pub depth: SearchDepth,
    pub searched_node_count: NodeCount,
    pub selected_depth: SearchPly,
    pub searched_time_ms: MilliSeconds,
    pub hash_utilization_permill: usize,
}

pub struct SearchEngine {
    transposition_table: TranspositionTable,
    counter_move_table: CounterMoveTable,
    killer_table: KillerTable,
    quiet_history_table: HistoryTable,
    searched_node_count: NodeCount,
    selected_depth: SearchPly,
    search_start_time: Instant,
    allowed_search_time: Duration,
    aborted: bool,
    force_stopped: Arc<AtomicBool>,
}

impl SearchEngine {
    pub fn new(transposition_table: TranspositionTable) -> Self {
        SearchEngine {
            transposition_table,
            counter_move_table: [[[EMPTY_CHESS_MOVE; CHESS_SQUARE_COUNT]; CHESS_SQUARE_COUNT];
                PLAYER_COUNT],
            killer_table: [[[EMPTY_CHESS_MOVE; KILLER_COUNT]; MAX_PV_LENGTH]; PLAYER_COUNT],
            quiet_history_table: [[0; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT],

            searched_node_count: 0,
            selected_depth: 0,

            search_start_time: Instant::now(),
            allowed_search_time: Duration::from_millis(0),

            aborted: false,
            force_stopped: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn reset_game(&mut self) {
        self.transposition_table.clear();
    }

    pub fn set_hash_size(&mut self, hash_size: usize) {
        self.transposition_table.re_size(hash_size);
    }

    pub fn search_best_move(
        &mut self,
        chess_position: &mut ChessPosition,
        allowed_time: Duration,
        force_stopped: Arc<AtomicBool>,
        show_output: bool,
    ) -> ChessMove {
        self.search_start_time = Instant::now();
        self.allowed_search_time = allowed_time;
        self.force_stopped = force_stopped;

        let mut best_move = EMPTY_CHESS_MOVE;
        let mut depth = 1;
        let mut alpha = -MATE_SCORE;
        let mut beta = MATE_SCORE;

        self.searched_node_count = 0;
        self.selected_depth = 0;
        self.aborted = false;
        self.counter_move_table =
            [[[EMPTY_CHESS_MOVE; CHESS_SQUARE_COUNT]; CHESS_SQUARE_COUNT]; PLAYER_COUNT];
        self.killer_table = [[[EMPTY_CHESS_MOVE; KILLER_COUNT]; MAX_PV_LENGTH]; PLAYER_COUNT];
        self.quiet_history_table = [[0; CHESS_SQUARE_COUNT]; PIECE_TYPE_COUNT];

        let soft_stop_time = allowed_time / 2;
        let in_check = is_in_check(chess_position, chess_position.player);

        let (has_forced_move, forced_move) = check_for_forced_move(chess_position, in_check);

        if has_forced_move {
            return forced_move;
        }

        loop {
            if self.search_start_time.elapsed() >= soft_stop_time {
                break;
            }

            let score = self.ab_search(chess_position, alpha, beta, in_check, depth, 0);

            if self.aborted {
                break;
            }

            if score <= alpha {
                alpha = -MATE_SCORE;
                continue;
            }

            if score >= beta {
                beta = MATE_SCORE;
                continue;
            }

            let mut principal_variation = Vec::new();
            self.retrieve_principal_variation(chess_position, &mut principal_variation);

            if !principal_variation.is_empty() {
                best_move = principal_variation[0];
            }

            if show_output {
                self.print_search_info(score, depth, &principal_variation);
            }

            if score.abs() > TERMINATE_SCORE {
                break;
            }

            depth += 1;
            alpha = score - WINDOW_SIZE;
            beta = score + WINDOW_SIZE;
        }

        best_move
    }

    fn ab_search(
        &mut self,
        chess_position: &mut ChessPosition,
        mut alpha: Score,
        beta: Score,
        in_check: bool,
        mut depth: SearchDepth,
        ply: SearchPly,
    ) -> Score {
        self.searched_node_count += 1;

        if self.force_stopped.load(Ordering::Relaxed)
            || self.search_start_time.elapsed() >= self.allowed_search_time
        {
            self.aborted = true;
            return alpha;
        }

        if ply > 0 && chess_position.is_draw() {
            return 0;
        }

        if in_check {
            depth += 1;
        }

        let safety_check = chess_position.white_all_bitboard * chess_position.black_all_bitboard;
        let hash_age = chess_position.full_move_count;

        let mut best_move = EMPTY_CHESS_MOVE;
        let mut valid_move_count = 0;
        let mut alpha_raised = false;

        let mut hash_entry = self.lookup_hash(chess_position.hash_key, safety_check);

        if hash_entry.is_none() && depth > IID_MIN_DEPTH && beta - alpha > 1 {
            self.ab_search(chess_position, alpha, beta, in_check, IID_SEARCH_DEPTH, ply);

            hash_entry = self.lookup_hash(chess_position.hash_key, safety_check);
        }

        if let Some(entry) = hash_entry {
            if entry.depth >= depth && entry.score != 0 {
                match entry.flag {
                    HashFlag::LowBound => {
                        if entry.score >= beta {
                            return entry.score;
                        }
                    }
                    HashFlag::HighBound => {
                        if entry.score <= alpha {
                            return entry.score;
                        }
                    }
                    HashFlag::Exact => {
                        let score = entry.score;
                        if score.abs() < TERMINATE_SCORE {
                            return score;
                        }
                    }
                }
            }

            if entry.depth > 0 && !entry.chess_move.is_empty() {
                best_move = entry.chess_move;
                valid_move_count += 1;
            }
        }

        if depth == 0 {
            return self.q_search(chess_position, alpha, beta, ply);
        }

        if !in_check && beta - alpha == 1 && beta > -TERMINATE_SCORE {
            let static_eval = chess_position.get_static_score();

            if depth < FUTILITY_PRUNING_MAX_DEPTH && static_eval - FUTILITY_PRUNING_MARGINS[depth as usize] > beta {
                return beta;
            }

            if depth >= NULL_MOVE_PRUNING_MIN_DEPTH && static_eval >= beta {
                let saved_enpassant_square = chess_position.make_null_move();

                let scout_score = -self.ab_search(chess_position, -beta, 1-beta, false, depth - NULL_MOVE_PRUNING_DEPTH_REDUCTION - 1, ply + 1);

                chess_position.unmake_null_move(saved_enpassant_square);

                if scout_score >= beta && scout_score != 0 && scout_score < TERMINATE_SCORE {
                    return beta;
                }
            }
        }

        if !best_move.is_empty() {
            let saved_state = chess_position.make_move(&best_move);

            let score = -self.ab_search(
                chess_position,
                -beta,
                -alpha,
                is_in_check(chess_position, chess_position.player),
                depth - 1,
                ply + 1,
            );

            chess_position.unmake_move(&best_move, saved_state);

            if self.aborted {
                return alpha;
            }

            if score >= beta {
                self.update_hash(&TableEntry {
                    key: chess_position.hash_key,
                    safety_check,
                    score: beta,
                    depth,
                    hash_age,
                    flag: HashFlag::LowBound,
                    chess_move: best_move,
                });

                if ply > 0 {
                    let last_move = chess_position.get_last_move();
                    self.counter_move_table[(chess_position.player ^ BLACK) as usize]
                        [last_move.from_square][last_move.to_square] = best_move;
                }

                return beta;
            }

            if score > alpha {
                alpha = score;
                alpha_raised = true;
            }
        }

        if self.searched_node_count & HISTORY_DECAY_INTERVAL == 0 {
            for piece in WP..=BK {
                let piece = piece as usize;

                for chess_square in A1..=H8 {
                    self.quiet_history_table[piece][chess_square] >>=
                        HISTORY_SCORE_SHIFT_FACTOR_DECAY;
                }
            }
        }

        let mut captures_and_promotions = self.sort_captures_and_promotions(
            chess_position,
            generate_captures_and_promotions(chess_position),
            ply,
        );

        while let Some(sortable_chess_move) = captures_and_promotions.pop() {
            let chess_move = sortable_chess_move.chess_move;

            if chess_move == best_move {
                continue;
            }

            let saved_state = chess_position.make_move(&chess_move);

            if is_invalid_position(chess_position) {
                chess_position.unmake_move(&chess_move, saved_state);
                continue;
            }

            valid_move_count += 1;

            let gives_check = is_in_check(chess_position, chess_position.player);

            let mut score;

            if valid_move_count == 1 {
                score = -self.ab_search(
                    chess_position,
                    -beta,
                    -alpha,
                    gives_check,
                    depth - 1,
                    ply + 1,
                );
            } else {
                let depth_reduction = if !gives_check && depth > 1 && sortable_chess_move.reducable {
                    u16_sqrt(depth).min(depth - 1)
                } else {
                    0
                };

                score = -self.ab_search(
                    chess_position,
                    -alpha - 1,
                    -alpha,
                    gives_check,
                    depth - depth_reduction - 1,
                    ply + 1,
                );

                if score > alpha && (score < beta || depth_reduction != 0) {
                    score = -self.ab_search(
                        chess_position,
                        -beta,
                        -alpha,
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
                    safety_check: chess_position.white_all_bitboard
                        * chess_position.black_all_bitboard,
                    score: beta,
                    depth,
                    hash_age,
                    flag: HashFlag::LowBound,
                    chess_move,
                });

                if ply > 0 {
                    let last_move = chess_position.get_last_move();
                    self.counter_move_table[(chess_position.player ^ BLACK) as usize]
                        [last_move.from_square][last_move.to_square] = best_move;
                }

                return beta;
            }

            if score > alpha {
                alpha = score;
                best_move = chess_move;
                alpha_raised = true;
            }
        }

        let mut quiet_chess_moves = self.sort_quiet_moves(
            chess_position,
            generate_quiet_moves(chess_position, in_check),
            ply,
        );

        let mut searched_quiet_chess_moves: Vec<ChessMove> = Vec::new();

        while let Some(sortable_chess_move) = quiet_chess_moves.pop() {
            let chess_move = sortable_chess_move.chess_move;

            if chess_move == best_move {
                continue;
            }

            let saved_state = chess_position.make_move(&chess_move);

            if is_invalid_position(chess_position) {
                chess_position.unmake_move(&chess_move, saved_state);
                continue;
            }

            valid_move_count += 1;

            let gives_check = is_in_check(chess_position, chess_position.player);

            let mut score;

            if valid_move_count == 1 {
                score = -self.ab_search(
                    chess_position,
                    -beta,
                    -alpha,
                    gives_check,
                    depth - 1,
                    ply + 1,
                );
            } else {
                let depth_reduction = if !gives_check && depth > 1 && sortable_chess_move.reducable {
                    u16_sqrt(depth).min(depth - 1)
                } else {
                    0
                };

                score = -self.ab_search(
                    chess_position,
                    -alpha - 1,
                    -alpha,
                    gives_check,
                    depth - depth_reduction - 1,
                    ply + 1,
                );

                if score > alpha && (score < beta || depth_reduction != 0) {
                    score = -self.ab_search(
                        chess_position,
                        -beta,
                        -alpha,
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
                    safety_check: chess_position.white_all_bitboard
                        * chess_position.black_all_bitboard,
                    score: beta,
                    depth,
                    hash_age,
                    flag: HashFlag::LowBound,
                    chess_move,
                });

                if ply > 0 {
                    let last_move = chess_position.get_last_move();
                    self.counter_move_table[(chess_position.player ^ BLACK) as usize]
                        [last_move.from_square][last_move.to_square] = best_move;
                }

                if ply < MAX_PV_LENGTH {
                    self.killer_table[chess_position.player as usize][ply][KILLER_INDEX_CUTOFF] =
                        chess_move;
                }

                let history_score_change = (depth as Score) << HISTORY_SCORE_SHIFT_FACTOR_CUTOFF;

                self.quiet_history_table[chess_position.board[chess_move.from_square] as usize]
                    [chess_move.to_square] += history_score_change;

                for searched_quiet_chess_move in &searched_quiet_chess_moves {
                    self.quiet_history_table
                        [chess_position.board[searched_quiet_chess_move.from_square] as usize]
                        [searched_quiet_chess_move.to_square] -= history_score_change;
                }

                return beta;
            }

            if score > alpha {
                alpha = score;
                best_move = chess_move;
                alpha_raised = true;

                if ply < MAX_PV_LENGTH {
                    self.killer_table[chess_position.player as usize][ply][KILLER_INDEX_RAISE] =
                        chess_move;
                }

                self.quiet_history_table[chess_position.board[chess_move.from_square] as usize]
                    [chess_move.to_square] += (depth as Score) << HISTORY_SCORE_SHIFT_FACTOR_RAISE;
            }

            searched_quiet_chess_moves.push(chess_move);
        }

        if valid_move_count == 0 {
            return if in_check {
                -MATE_SCORE + ply as Score
            } else {
                0
            };
        }

        if alpha_raised {
            self.update_hash(&TableEntry {
                key: chess_position.hash_key,
                safety_check,
                score: alpha,
                depth,
                hash_age,
                flag: HashFlag::Exact,
                chess_move: best_move,
            });
        } else if !best_move.is_empty() {
            self.update_hash(&TableEntry {
                key: chess_position.hash_key,
                safety_check,
                score: alpha,
                depth,
                hash_age,
                flag: HashFlag::HighBound,
                chess_move: best_move,
            });
        }

        alpha
    }

    fn q_search(
        &mut self,
        chess_position: &mut ChessPosition,
        mut alpha: Score,
        beta: Score,
        ply: SearchPly,
    ) -> Score {
        self.searched_node_count += 1;

        if ply > self.selected_depth {
            self.selected_depth = ply;
        }

        if self.search_start_time.elapsed() >= self.allowed_search_time {
            self.aborted = true;
            return alpha;
        }

        let static_eval = chess_position.get_static_score();

        if static_eval >= beta {
            return beta;
        }

        if static_eval > alpha {
            alpha = static_eval;
        }

        let mut captures_and_promotions = self.sort_captures_and_promotions(
            chess_position,
            generate_captures_and_promotions(chess_position),
            ply,
        );

        while let Some(sortable_chess_move) = captures_and_promotions.pop() {
            let chess_move = sortable_chess_move.chess_move;

            let saved_state = chess_position.make_move(&chess_move);

            if is_invalid_position(chess_position) {
                chess_position.unmake_move(&chess_move, saved_state);
                continue;
            }

            let score = -self.q_search(chess_position, -beta, -alpha, ply + 1);

            chess_position.unmake_move(&chess_move, saved_state);

            if self.aborted {
                return alpha;
            }

            if score >= beta {
                return beta;
            }

            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }

    fn sort_captures_and_promotions(
        &self,
        chess_position: &mut ChessPosition,
        chess_moves: Vec<ChessMove>,
        ply: SearchPly,
    ) -> BinaryHeap<SortableChessMove> {
        let mut sorted_moves = BinaryHeap::new();
        let mut max_exchange_score = -MATE_SCORE;

        let mut counter_move = EMPTY_CHESS_MOVE;
        let counter_move_to_last_move = if ply > 0 {
            let last_move = chess_position.get_last_move();
            self.counter_move_table[chess_position.player as usize][last_move.from_square]
                [last_move.to_square]
        } else {
            EMPTY_CHESS_MOVE
        };

        for chess_move in chess_moves {
            if chess_move == counter_move_to_last_move {
                counter_move = chess_move;
                continue;
            }

            let mut sort_score = SORTING_PIECE_VALS
                [chess_position.board[chess_move.to_square] as usize]
                - SORTING_PIECE_VALS[chess_position.board[chess_move.from_square] as usize]
                + SORTING_PIECE_VALS[chess_move.promotion_piece as usize];

            if sort_score <= 0 {
                sort_score = static_exchange_evaluate(chess_position, &chess_move, 0);
            }

            if sort_score > max_exchange_score {
                max_exchange_score = sort_score;
            }

            sorted_moves.push(SortableChessMove {
                chess_move,
                sort_score,
                reducable: sort_score < 0,
            });
        }

        if !counter_move.is_empty() {
            sorted_moves.push(SortableChessMove {
                chess_move: counter_move,
                sort_score: max_exchange_score + 1,
                reducable: false,
            });
        }

        sorted_moves
    }

    fn sort_quiet_moves(
        &self,
        chess_position: &mut ChessPosition,
        chess_moves: Vec<ChessMove>,
        ply: SearchPly,
    ) -> BinaryHeap<SortableChessMove> {
        let mut sorted_moves = BinaryHeap::new();
        let mut max_history_score = -MATE_SCORE;

        let mut primary_killer_move = EMPTY_CHESS_MOVE;
        let mut secondary_killer_move = EMPTY_CHESS_MOVE;
        let mut counter_move = EMPTY_CHESS_MOVE;

        let counter_move_to_last_move = if ply > 0 {
            let last_move = chess_position.get_last_move();
            self.counter_move_table[chess_position.player as usize][last_move.from_square]
                [last_move.to_square]
        } else {
            EMPTY_CHESS_MOVE
        };

        let primary_killer_in_ply = if ply < MAX_PV_LENGTH {
            self.killer_table[chess_position.player as usize][ply][KILLER_INDEX_CUTOFF]
        } else {
            EMPTY_CHESS_MOVE
        };

        let secondary_killer_in_play = if ply < MAX_PV_LENGTH {
            self.killer_table[chess_position.player as usize][ply][KILLER_INDEX_RAISE]
        } else {
            EMPTY_CHESS_MOVE
        };

        for chess_move in chess_moves {
            if chess_move == counter_move_to_last_move {
                counter_move = chess_move;
                continue;
            } else if chess_move == primary_killer_in_ply {
                primary_killer_move = chess_move;
                continue;
            } else if chess_move == secondary_killer_in_play {
                secondary_killer_move = chess_move;
                continue;
            }

            let history_score = self.quiet_history_table
                [chess_position.board[chess_move.from_square] as usize][chess_move.to_square];

            if history_score > max_history_score {
                max_history_score = history_score;
            }

            sorted_moves.push(SortableChessMove {
                chess_move,
                sort_score: history_score,
                reducable: true,
            });
        }

        if !counter_move.is_empty() {
            sorted_moves.push(SortableChessMove {
                chess_move: counter_move,
                sort_score: max_history_score + 3,
                reducable: false,
            });
        }

        if !primary_killer_move.is_empty() {
            sorted_moves.push(SortableChessMove {
                chess_move: primary_killer_move,
                sort_score: max_history_score + 2,
                reducable: false,
            });
        }

        if !secondary_killer_move.is_empty() {
            sorted_moves.push(SortableChessMove {
                chess_move: secondary_killer_move,
                sort_score: max_history_score + 1,
                reducable: false,
            });
        }

        sorted_moves
    }

    pub fn perft(
        &mut self,
        chess_position: &mut ChessPosition,
        depth: SearchDepth,
        start_time: &Instant,
    ) -> ChessMoveCount {
        if depth == 0 {
            return 1;
        }

        let mut chess_move_count = 0;

        let captures_and_promotions = generate_captures_and_promotions(chess_position);

        for chess_move in &captures_and_promotions {
            let saved_state = chess_position.make_move(chess_move);

            if is_invalid_position(chess_position) {
                chess_position.unmake_move(chess_move, saved_state);
                continue;
            }

            let next_perft = self.perft_helper(chess_position, depth - 1);

            println!("{}: {}", format_chess_move(chess_move), next_perft);

            chess_move_count += next_perft;
            chess_position.unmake_move(chess_move, saved_state);
        }

        let quiet_moves = generate_quiet_moves(
            chess_position,
            is_in_check(chess_position, chess_position.player),
        );

        for chess_move in &quiet_moves {
            let saved_state = chess_position.make_move(chess_move);

            if is_invalid_position(chess_position) {
                chess_position.unmake_move(chess_move, saved_state);
                continue;
            }

            let next_perft = self.perft_helper(chess_position, depth - 1);

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

    fn perft_helper(
        &mut self,
        chess_position: &mut ChessPosition,
        depth: SearchDepth,
    ) -> ChessMoveCount {
        if depth == 0 {
            return 1;
        }

        if let Some(entry) = self.lookup_hash(
            chess_position.hash_key,
            chess_position.white_all_bitboard * chess_position.black_all_bitboard,
        ) {
            if entry.depth == depth {
                return entry.score as ChessMoveCount;
            }
        }

        let mut chess_move_count = 0;

        let captures_and_promotions = generate_captures_and_promotions(chess_position);

        for chess_move in &captures_and_promotions {
            let saved_state = chess_position.make_move(chess_move);

            if is_invalid_position(chess_position) {
                chess_position.unmake_move(chess_move, saved_state);
                continue;
            }

            let next_perft = self.perft_helper(chess_position, depth - 1);

            chess_move_count += next_perft;
            chess_position.unmake_move(chess_move, saved_state);
        }

        let quiet_moves = generate_quiet_moves(
            chess_position,
            is_in_check(chess_position, chess_position.player),
        );

        for chess_move in &quiet_moves {
            let saved_state = chess_position.make_move(chess_move);

            if is_invalid_position(chess_position) {
                chess_position.unmake_move(chess_move, saved_state);
                continue;
            }

            let next_perft = self.perft_helper(chess_position, depth - 1);

            chess_move_count += next_perft;
            chess_position.unmake_move(chess_move, saved_state);
        }

        self.update_hash(&TableEntry {
            key: chess_position.hash_key,
            safety_check: chess_position.white_all_bitboard * chess_position.black_all_bitboard,
            score: chess_move_count as Score,
            depth,
            hash_age: 0,
            flag: HashFlag::Exact,
            chess_move: EMPTY_CHESS_MOVE,
        });

        chess_move_count
    }

    fn retrieve_principal_variation(
        &mut self,
        chess_position: &mut ChessPosition,
        principal_variation: &mut Vec<ChessMove>,
    ) {
        if principal_variation.len() > MAX_PV_LENGTH {
            return;
        }

        if let Some(entry) = self.lookup_hash(
            chess_position.hash_key,
            chess_position.white_all_bitboard * chess_position.black_all_bitboard,
        ) {
            let chess_move = entry.chess_move;

            if !chess_move.is_empty() {
                principal_variation.push(chess_move);

                let saved_state = chess_position.make_move(&chess_move);
                self.retrieve_principal_variation(chess_position, principal_variation);
                chess_position.unmake_move(&chess_move, saved_state);
            }
        }
    }

    #[inline(always)]
    fn lookup_hash(&self, hash_key: HashKey, safety_check: BitBoard) -> Option<&TableEntry> {
        self.transposition_table.get(hash_key, safety_check)
    }

    #[inline(always)]
    fn update_hash(&mut self, table_entry: &TableEntry) {
        self.transposition_table.set(table_entry);
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
                searched_node_count: self.searched_node_count,
                selected_depth: self.selected_depth,
                searched_time_ms: self.search_start_time.elapsed().as_millis() as MilliSeconds,
                hash_utilization_permill: self.transposition_table.get_utilization_permill(),
            },
            principal_variation,
        );
    }
}

fn static_exchange_evaluate(
    chess_position: &mut ChessPosition,
    chess_move: &ChessMove,
    ply: SearchDepth,
) -> Score {
    let attack_square = chess_move.to_square;
    let initial_gain = SORTING_PIECE_VALS[chess_position.board[attack_square] as usize]
        + SORTING_PIECE_VALS[chess_move.promotion_piece as usize];

    let saved_state = chess_position.make_move(chess_move);
    let attackers = get_least_valued_attackers_to_chess_square(chess_position, attack_square);

    if attackers.is_empty() {
        chess_position.unmake_move(chess_move, saved_state);
        return initial_gain;
    }

    let mut max_opponent_gain = 0;

    for attacker in attackers {
        let opponent_gain = static_exchange_evaluate(chess_position, &attacker, ply + 1);

        if opponent_gain > max_opponent_gain {
            max_opponent_gain = opponent_gain;
        }
    }

    chess_position.unmake_move(chess_move, saved_state);

    if ply == 0 {
        initial_gain - max_opponent_gain
    } else {
        0.max(initial_gain - max_opponent_gain)
    }
}

fn check_for_forced_move(chess_position: &mut ChessPosition, in_check: bool) -> (bool, ChessMove) {
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
