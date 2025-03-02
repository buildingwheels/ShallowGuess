use crate::bit_masks::{EMPTY_MASK, SQUARE_MASKS};
use crate::def::{
    A1, A8, BK, BLACK, BP, BR, CASTLING_FLAG_BLACK, CASTLING_FLAG_BLACK_KING_SIDE,
    CASTLING_FLAG_BLACK_QUEEN_SIDE, CASTLING_FLAG_EMPTY, CASTLING_FLAG_WHITE,
    CASTLING_FLAG_WHITE_KING_SIDE, CASTLING_FLAG_WHITE_QUEEN_SIDE, CHESS_FILE_COUNT, D1, D8, F1,
    F8, FILE_H, G1, G8, H1, H8, NO_PIECE, NO_SQUARE, WHITE, WK, WP, WR,
};
use crate::eval::MATERIAL_SCORE;
use crate::fen::{
    fen_str_constants, get_chess_piece_from_char, get_chess_square_from_chars,
    CHESS_PIECE_TO_CHESS_PIECE_CHAR_MAP, CHESS_SQUARE_INDEX_TO_STR_MAP,
};
use crate::network::Network;
use crate::types::{
    CastlingFlag, ChessMove, ChessMoveCount, ChessMoveType, ChessSquare, HashKey, Score,
};
use crate::util::{char_to_digit, digit_to_char, get_file};
use crate::zobrist::{CASTLING_FLAG_HASH, ENPASSANT_SQUARE_HASH, PIECE_SQUARE_HASH, PLAYER_HASH};
use crate::{
    def::{CHESS_SQUARE_COUNT, PIECE_TYPE_COUNT},
    types::{BitBoard, ChessPiece, Player},
};

const FIFTY_MOVES: ChessMoveCount = 100;

pub struct RecoverablePositionState {
    captured_piece: ChessPiece,
    saved_enpassant_square: ChessSquare,
    saved_castling_flag: CastlingFlag,
    saved_white_all_bitboard: BitBoard,
    saved_black_all_bitboard: BitBoard,
    saved_half_move_count: ChessMoveCount,
}

pub struct ChessPosition {
    pub board: [ChessPiece; CHESS_SQUARE_COUNT],
    pub bitboards: [BitBoard; PIECE_TYPE_COUNT],
    pub network: Network,
    pub white_all_bitboard: BitBoard,
    pub black_all_bitboard: BitBoard,
    pub player: Player,
    pub enpassant_square: ChessSquare,
    pub castling_flag: CastlingFlag,
    pub white_king_square: ChessSquare,
    pub black_king_square: ChessSquare,
    pub half_move_count: ChessMoveCount,
    pub full_move_count: ChessMoveCount,
    pub hash_key: HashKey,

    material_score: Score,
    hash_key_history: Vec<HashKey>,
    chess_move_history: Vec<ChessMove>,
}

impl ChessPosition {
    pub fn new(network: Network) -> Self {
        ChessPosition {
            board: [NO_PIECE; CHESS_SQUARE_COUNT],
            bitboards: [EMPTY_MASK; PIECE_TYPE_COUNT],
            network,
            white_all_bitboard: EMPTY_MASK,
            black_all_bitboard: EMPTY_MASK,
            player: WHITE,
            enpassant_square: NO_SQUARE,
            castling_flag: CASTLING_FLAG_EMPTY,
            white_king_square: 0,
            black_king_square: 0,
            half_move_count: 0,
            full_move_count: 0,
            hash_key: 0,
            material_score: 0,

            hash_key_history: Vec::new(),
            chess_move_history: Vec::new(),
        }
    }

    pub fn set_from_fen(&mut self, fen: &str) {
        self.hash_key = 0;
        self.hash_key_history.clear();
        self.chess_move_history.clear();

        self.material_score = 0;

        self.board = [NO_PIECE; CHESS_SQUARE_COUNT];
        self.bitboards = [EMPTY_MASK; PIECE_TYPE_COUNT];
        self.network.clear_accumulated_layer();
        self.white_all_bitboard = EMPTY_MASK;
        self.black_all_bitboard = EMPTY_MASK;
        self.castling_flag = 0;

        let mut fen_segments = fen.trim().split(fen_str_constants::SPLITTER);
        let squares_str = fen_segments.next().unwrap();
        let player_str = fen_segments.next().unwrap();
        let castling_flag_str = fen_segments.next().unwrap();
        let enp_sqr_str = fen_segments.next().unwrap();
        let half_move_str = fen_segments.next().unwrap();
        let full_move_str = fen_segments.next().unwrap();

        self.player = if player_str == fen_str_constants::PLAYER_WHITE {
            WHITE
        } else {
            BLACK
        };

        self.hash_key ^= PLAYER_HASH[self.player as usize];

        self.half_move_count = half_move_str.parse::<ChessMoveCount>().unwrap();
        self.full_move_count = full_move_str.parse::<ChessMoveCount>().unwrap();

        if castling_flag_str.contains(fen_str_constants::CASTLING_FLAG_WHITE_KING_SIDE) {
            self.castling_flag |= CASTLING_FLAG_WHITE_KING_SIDE;
        }
        if castling_flag_str.contains(fen_str_constants::CASTLING_FLAG_WHITE_QUEEN_SIDE) {
            self.castling_flag |= CASTLING_FLAG_WHITE_QUEEN_SIDE;
        }
        if castling_flag_str.contains(fen_str_constants::CASTLING_FLAG_BLACK_KING_SIDE) {
            self.castling_flag |= CASTLING_FLAG_BLACK_KING_SIDE;
        }
        if castling_flag_str.contains(fen_str_constants::CASTLING_FLAG_BLACK_QUEEN_SIDE) {
            self.castling_flag |= CASTLING_FLAG_BLACK_QUEEN_SIDE;
        }

        self.hash_key ^= CASTLING_FLAG_HASH[self.castling_flag as usize];

        if enp_sqr_str != fen_str_constants::NA_STR {
            let mut enp_sqr_str_iter = enp_sqr_str.chars().into_iter();
            self.enpassant_square = get_chess_square_from_chars(
                enp_sqr_str_iter.next().unwrap(),
                enp_sqr_str_iter.next().unwrap(),
            );
            self.hash_key ^= ENPASSANT_SQUARE_HASH[self.enpassant_square];
        } else {
            self.enpassant_square = NO_SQUARE;
        }

        let mut square_chars = squares_str.chars();
        let mut current_square = A8;

        loop {
            let next_char_option = square_chars.next();

            if next_char_option.is_none() {
                break;
            }

            let next_char = next_char_option.unwrap();

            if next_char == fen_str_constants::RANK_BREAK {
                current_square -= CHESS_FILE_COUNT * 2;
                continue;
            }

            if next_char.is_digit(10) {
                current_square += char_to_digit(next_char);
                continue;
            }

            let piece = get_chess_piece_from_char(next_char);
            self.board[current_square] = piece;
            self.bitboards[piece as usize] |= SQUARE_MASKS[current_square];
            self.hash_key ^= PIECE_SQUARE_HASH[piece as usize][current_square];
            self.network.add(piece, current_square);
            self.material_score += MATERIAL_SCORE[piece as usize];

            if piece < BP {
                self.white_all_bitboard |= SQUARE_MASKS[current_square];

                if piece == WK {
                    self.white_king_square = current_square;
                }
            } else {
                self.black_all_bitboard |= SQUARE_MASKS[current_square];

                if piece == BK {
                    self.black_king_square = current_square;
                }
            }

            current_square += 1;
        }
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        let mut current_square = A8;
        let mut empty_square_count = 0;

        loop {
            let chess_piece = self.board[current_square];

            if chess_piece == NO_PIECE {
                empty_square_count += 1;
            } else {
                if empty_square_count != 0 {
                    fen.push(digit_to_char(empty_square_count));
                    empty_square_count = 0;
                }

                fen.push(CHESS_PIECE_TO_CHESS_PIECE_CHAR_MAP[chess_piece as usize]);
            }

            if current_square == H1 {
                if empty_square_count != 0 {
                    fen.push(digit_to_char(empty_square_count));
                }

                break;
            }

            if get_file(current_square) == FILE_H {
                current_square -= CHESS_FILE_COUNT * 2 - 1;

                if empty_square_count != 0 {
                    fen.push(digit_to_char(empty_square_count));
                    empty_square_count = 0;
                }

                fen.push(fen_str_constants::RANK_BREAK);
                continue;
            }

            current_square += 1;
        }

        fen.push_str(fen_str_constants::SPLITTER);

        if self.player == WHITE {
            fen.push_str(fen_str_constants::PLAYER_WHITE);
        } else {
            fen.push_str(fen_str_constants::PLAYER_BLACK);
        }

        fen.push_str(fen_str_constants::SPLITTER);

        if self.castling_flag == CASTLING_FLAG_EMPTY {
            fen.push_str(fen_str_constants::NA_STR);
        } else {
            if (self.castling_flag & CASTLING_FLAG_WHITE_KING_SIDE) != 0 {
                fen.push_str(fen_str_constants::CASTLING_FLAG_WHITE_KING_SIDE);
            }
            if (self.castling_flag & CASTLING_FLAG_WHITE_QUEEN_SIDE) != 0 {
                fen.push_str(fen_str_constants::CASTLING_FLAG_WHITE_QUEEN_SIDE);
            }
            if (self.castling_flag & CASTLING_FLAG_BLACK_KING_SIDE) != 0 {
                fen.push_str(fen_str_constants::CASTLING_FLAG_BLACK_KING_SIDE);
            }
            if (self.castling_flag & CASTLING_FLAG_BLACK_QUEEN_SIDE) != 0 {
                fen.push_str(fen_str_constants::CASTLING_FLAG_BLACK_QUEEN_SIDE);
            }
        }

        fen.push_str(fen_str_constants::SPLITTER);

        if self.enpassant_square == NO_SQUARE {
            fen.push_str(fen_str_constants::NA_STR);
        } else {
            fen.push_str(CHESS_SQUARE_INDEX_TO_STR_MAP[self.enpassant_square]);
        }

        fen.push_str(&format!(" {} ", self.half_move_count));
        fen.push_str(&format!("{}", self.full_move_count));

        fen
    }

    pub fn is_draw(&self) -> bool {
        if self.half_move_count >= FIFTY_MOVES {
            return true;
        }

        let history_len = self.hash_key_history.len();

        for search_index in 0..history_len {
            if self.hash_key == self.hash_key_history[search_index] {
                return true;
            }
        }

        false
    }

    pub fn get_last_move(&self) -> &ChessMove {
        self.chess_move_history.last().unwrap()
    }

    pub fn get_material_score(&self) -> Score {
        if self.player == WHITE {
            self.material_score
        } else {
            -self.material_score
        }
    }

    pub fn make_null_move(&mut self) -> ChessSquare {
        self.modify_hash();
        let saved_enpassant_square = self.enpassant_square;
        self.enpassant_square = NO_SQUARE;
        self.player ^= BLACK;
        self.modify_hash();

        return saved_enpassant_square;
    }

    pub fn unmake_null_move(&mut self, saved_enpassant_square: ChessSquare) {
        self.modify_hash();
        self.enpassant_square = saved_enpassant_square;
        self.player ^= BLACK;
        self.modify_hash();
    }

    pub fn make_move(&mut self, chess_move: &ChessMove) -> RecoverablePositionState {
        let saved_enpassant_square = self.enpassant_square;
        let saved_castling_flag = self.castling_flag;
        let saved_white_all_bitboard = self.white_all_bitboard;
        let saved_black_all_bitboard = self.black_all_bitboard;
        let saved_half_move_count = self.half_move_count;

        self.chess_move_history.push(*chess_move);
        self.hash_key_history.push(self.hash_key);
        self.modify_hash();

        self.full_move_count += 1;
        self.half_move_count += 1;

        let mut captured_piece = NO_PIECE;

        self.enpassant_square = NO_SQUARE;

        let from_square = chess_move.from_square;
        let to_square = chess_move.to_square;
        let moving_piece = self.board[from_square];

        if moving_piece == WP || moving_piece == BP {
            self.half_move_count = 0;
        }

        let from_square_mask = SQUARE_MASKS[from_square];
        let to_square_mask = SQUARE_MASKS[to_square];

        match chess_move.move_type {
            ChessMoveType::Regular => {
                captured_piece = self.board[to_square];

                self.board[from_square] = NO_PIECE;
                self.board[to_square] = moving_piece;

                self.bitboards[moving_piece as usize] ^= from_square_mask | to_square_mask;

                self.network.remove(moving_piece, from_square);
                self.network.add(moving_piece, to_square);

                self.hash_key ^= PIECE_SQUARE_HASH[moving_piece as usize][from_square];
                self.hash_key ^= PIECE_SQUARE_HASH[moving_piece as usize][to_square];

                if moving_piece < BP {
                    self.white_all_bitboard ^= from_square_mask | to_square_mask;

                    if moving_piece == WK {
                        self.castling_flag &= !CASTLING_FLAG_WHITE;
                        self.white_king_square = to_square;
                    } else if moving_piece == WR {
                        if from_square == H1 {
                            self.castling_flag &= !CASTLING_FLAG_WHITE_KING_SIDE;
                        } else if from_square == A1 {
                            self.castling_flag &= !CASTLING_FLAG_WHITE_QUEEN_SIDE;
                        }
                    }
                } else {
                    self.black_all_bitboard ^= from_square_mask | to_square_mask;

                    if moving_piece == BK {
                        self.castling_flag &= !CASTLING_FLAG_BLACK;
                        self.black_king_square = to_square;
                    } else if moving_piece == BR {
                        if from_square == H8 {
                            self.castling_flag &= !CASTLING_FLAG_BLACK_KING_SIDE;
                        } else if from_square == A8 {
                            self.castling_flag &= !CASTLING_FLAG_BLACK_QUEEN_SIDE;
                        }
                    }
                }

                if captured_piece != NO_PIECE {
                    self.half_move_count = 0;
                    self.bitboards[captured_piece as usize] ^= to_square_mask;
                    self.network.remove(captured_piece, to_square);
                    self.material_score -= MATERIAL_SCORE[captured_piece as usize];

                    self.hash_key ^= PIECE_SQUARE_HASH[captured_piece as usize][to_square];

                    if captured_piece < BP {
                        self.white_all_bitboard ^= to_square_mask;

                        if captured_piece == WR {
                            if to_square == H1 {
                                self.castling_flag &= !CASTLING_FLAG_WHITE_KING_SIDE;
                            } else if to_square == A1 {
                                self.castling_flag &= !CASTLING_FLAG_WHITE_QUEEN_SIDE;
                            }
                        }
                    } else {
                        self.black_all_bitboard ^= to_square_mask;

                        if captured_piece == BR {
                            if to_square == H8 {
                                self.castling_flag &= !CASTLING_FLAG_BLACK_KING_SIDE;
                            } else if to_square == A8 {
                                self.castling_flag &= !CASTLING_FLAG_BLACK_QUEEN_SIDE;
                            }
                        }
                    }
                }
            }
            ChessMoveType::Promotion => {
                let promotion_piece = chess_move.promotion_piece;
                captured_piece = self.board[to_square];

                self.board[from_square] = NO_PIECE;
                self.board[to_square] = promotion_piece;

                self.bitboards[moving_piece as usize] ^= from_square_mask;
                self.bitboards[promotion_piece as usize] ^= to_square_mask;
                self.bitboards[captured_piece as usize] ^= to_square_mask;

                self.network.remove(moving_piece, from_square);
                self.network.add(promotion_piece, to_square);

                self.material_score -= MATERIAL_SCORE[moving_piece as usize];
                self.material_score += MATERIAL_SCORE[promotion_piece as usize];

                if moving_piece < BP {
                    self.white_all_bitboard ^= from_square_mask | to_square_mask;
                } else {
                    self.black_all_bitboard ^= from_square_mask | to_square_mask;
                }

                self.hash_key ^= PIECE_SQUARE_HASH[moving_piece as usize][from_square];
                self.hash_key ^= PIECE_SQUARE_HASH[promotion_piece as usize][to_square];

                if captured_piece != NO_PIECE {
                    self.network.remove(captured_piece, to_square);
                    self.hash_key ^= PIECE_SQUARE_HASH[captured_piece as usize][to_square];
                    self.material_score -= MATERIAL_SCORE[captured_piece as usize];

                    if captured_piece < BP {
                        self.white_all_bitboard ^= to_square_mask;

                        if captured_piece == WR {
                            if to_square == H1 {
                                self.castling_flag &= !CASTLING_FLAG_WHITE_KING_SIDE;
                            } else if to_square == A1 {
                                self.castling_flag &= !CASTLING_FLAG_WHITE_QUEEN_SIDE;
                            }
                        }
                    } else {
                        self.black_all_bitboard ^= to_square_mask;

                        if captured_piece == BR {
                            if to_square == H8 {
                                self.castling_flag &= !CASTLING_FLAG_BLACK_KING_SIDE;
                            } else if to_square == A8 {
                                self.castling_flag &= !CASTLING_FLAG_BLACK_QUEEN_SIDE;
                            }
                        }
                    }
                }
            }
            ChessMoveType::Castle => {
                self.half_move_count = 0;

                self.board[from_square] = NO_PIECE;
                self.board[to_square] = moving_piece;

                let moving_piece_mask = from_square_mask | to_square_mask;
                self.bitboards[moving_piece as usize] ^= moving_piece_mask;

                self.network.remove(moving_piece, from_square);
                self.network.add(moving_piece, to_square);

                self.hash_key ^= PIECE_SQUARE_HASH[moving_piece as usize][from_square];
                self.hash_key ^= PIECE_SQUARE_HASH[moving_piece as usize][to_square];

                if moving_piece == WK {
                    self.white_king_square = to_square;

                    if to_square == G1 {
                        self.board[H1] = NO_PIECE;
                        self.board[F1] = WR;
                        self.bitboards[WR as usize] ^= SQUARE_MASKS[H1] | SQUARE_MASKS[F1];
                        self.white_all_bitboard ^=
                            from_square_mask | to_square_mask | SQUARE_MASKS[H1] | SQUARE_MASKS[F1];
                        self.network.remove(WR, H1);
                        self.network.add(WR, F1);
                        self.hash_key ^= PIECE_SQUARE_HASH[WR as usize][H1];
                        self.hash_key ^= PIECE_SQUARE_HASH[WR as usize][F1];
                    } else {
                        self.board[A1] = NO_PIECE;
                        self.board[D1] = WR;
                        self.bitboards[WR as usize] ^= SQUARE_MASKS[A1] | SQUARE_MASKS[D1];
                        self.white_all_bitboard ^=
                            from_square_mask | to_square_mask | SQUARE_MASKS[A1] | SQUARE_MASKS[D1];
                        self.network.remove(WR, A1);
                        self.network.add(WR, D1);
                        self.hash_key ^= PIECE_SQUARE_HASH[WR as usize][A1];
                        self.hash_key ^= PIECE_SQUARE_HASH[WR as usize][D1];
                    }

                    self.castling_flag &= !CASTLING_FLAG_WHITE;
                } else if moving_piece == BK {
                    self.black_king_square = to_square;

                    if to_square == G8 {
                        self.board[H8] = NO_PIECE;
                        self.board[F8] = BR;
                        self.bitboards[BR as usize] ^= SQUARE_MASKS[H8] | SQUARE_MASKS[F8];
                        self.black_all_bitboard ^=
                            from_square_mask | to_square_mask | SQUARE_MASKS[H8] | SQUARE_MASKS[F8];
                        self.network.remove(BR, H8);
                        self.network.add(BR, F8);
                        self.hash_key ^= PIECE_SQUARE_HASH[BR as usize][H8];
                        self.hash_key ^= PIECE_SQUARE_HASH[BR as usize][F8];
                    } else {
                        self.board[A8] = NO_PIECE;
                        self.board[D8] = BR;
                        self.bitboards[BR as usize] ^= SQUARE_MASKS[A8] | SQUARE_MASKS[D8];
                        self.black_all_bitboard ^=
                            from_square_mask | to_square_mask | SQUARE_MASKS[A8] | SQUARE_MASKS[D8];
                        self.network.remove(BR, A8);
                        self.network.add(BR, D8);
                        self.hash_key ^= PIECE_SQUARE_HASH[BR as usize][A8];
                        self.hash_key ^= PIECE_SQUARE_HASH[BR as usize][D8];
                    }

                    self.castling_flag &= !CASTLING_FLAG_BLACK;
                }
            }
            ChessMoveType::EnPassant => {
                self.board[from_square] = NO_PIECE;
                self.board[to_square] = moving_piece;
                self.bitboards[moving_piece as usize] ^= from_square_mask | to_square_mask;
                self.network.remove(moving_piece, from_square);
                self.network.add(moving_piece, to_square);
                self.hash_key ^= PIECE_SQUARE_HASH[moving_piece as usize][from_square];
                self.hash_key ^= PIECE_SQUARE_HASH[moving_piece as usize][to_square];

                if moving_piece == WP {
                    let captured_square = to_square - CHESS_FILE_COUNT;
                    self.board[captured_square] = NO_PIECE;
                    self.bitboards[BP as usize] ^= SQUARE_MASKS[captured_square];
                    self.white_all_bitboard ^= from_square_mask | to_square_mask;
                    self.black_all_bitboard ^= SQUARE_MASKS[captured_square];
                    self.network.remove(BP, captured_square);
                    self.hash_key ^= PIECE_SQUARE_HASH[BP as usize][captured_square];
                    self.material_score -= MATERIAL_SCORE[BP as usize];
                } else {
                    let captured_square = to_square + CHESS_FILE_COUNT;
                    self.board[captured_square] = NO_PIECE;
                    self.bitboards[WP as usize] ^= SQUARE_MASKS[captured_square];
                    self.black_all_bitboard ^= from_square_mask | to_square_mask;
                    self.white_all_bitboard ^= SQUARE_MASKS[captured_square];
                    self.network.remove(WP, captured_square);
                    self.hash_key ^= PIECE_SQUARE_HASH[WP as usize][captured_square];
                    self.material_score -= MATERIAL_SCORE[WP as usize];
                }
            }
            ChessMoveType::CreateEnPassant => {
                self.board[from_square] = NO_PIECE;
                self.board[to_square] = moving_piece;
                self.bitboards[moving_piece as usize] ^= from_square_mask | to_square_mask;
                self.network.remove(moving_piece, from_square);
                self.network.add(moving_piece, to_square);
                self.hash_key ^= PIECE_SQUARE_HASH[moving_piece as usize][from_square];
                self.hash_key ^= PIECE_SQUARE_HASH[moving_piece as usize][to_square];

                if moving_piece == WP {
                    self.white_all_bitboard ^= from_square_mask | to_square_mask;
                    self.enpassant_square = to_square - CHESS_FILE_COUNT;
                } else {
                    self.black_all_bitboard ^= from_square_mask | to_square_mask;
                    self.enpassant_square = to_square + CHESS_FILE_COUNT;
                }
            }
        }

        self.player ^= BLACK;

        self.modify_hash();

        RecoverablePositionState {
            captured_piece,
            saved_enpassant_square,
            saved_castling_flag,
            saved_white_all_bitboard,
            saved_black_all_bitboard,
            saved_half_move_count,
        }
    }

    pub fn unmake_move(&mut self, chess_move: &ChessMove, saved_state: RecoverablePositionState) {
        self.chess_move_history.pop();

        self.full_move_count -= 1;
        self.half_move_count = saved_state.saved_half_move_count;
        self.white_all_bitboard = saved_state.saved_white_all_bitboard;
        self.black_all_bitboard = saved_state.saved_black_all_bitboard;

        self.hash_key = self.hash_key_history.pop().unwrap();

        self.enpassant_square = saved_state.saved_enpassant_square;
        self.castling_flag = saved_state.saved_castling_flag;
        self.player ^= BLACK;

        let from_square = chess_move.from_square;
        let to_square = chess_move.to_square;
        let moved_piece = self.board[to_square];
        let captured_piece = saved_state.captured_piece;

        let from_square_mask = SQUARE_MASKS[from_square];
        let to_square_mask = SQUARE_MASKS[to_square];

        match chess_move.move_type {
            ChessMoveType::Regular => {
                self.board[from_square] = moved_piece;
                self.board[to_square] = captured_piece;

                self.bitboards[moved_piece as usize] ^= from_square_mask | to_square_mask;

                self.network.add(moved_piece, from_square);
                self.network.remove(moved_piece, to_square);

                if captured_piece != NO_PIECE {
                    self.bitboards[captured_piece as usize] ^= to_square_mask;
                    self.network.add(captured_piece, to_square);
                    self.material_score += MATERIAL_SCORE[captured_piece as usize];
                }

                if moved_piece == WK {
                    self.white_king_square = from_square;
                } else if moved_piece == BK {
                    self.black_king_square = from_square;
                }
            }
            ChessMoveType::Promotion => {
                let moved_piece = if self.player == WHITE { WP } else { BP };
                self.board[from_square] = moved_piece;
                self.board[to_square] = NO_PIECE;
                self.bitboards[moved_piece as usize] ^= from_square_mask;

                let promotion_piece = chess_move.promotion_piece;
                self.bitboards[promotion_piece as usize] ^= to_square_mask;

                self.network.add(moved_piece, from_square);
                self.network.remove(promotion_piece, to_square);

                self.material_score += MATERIAL_SCORE[moved_piece as usize];
                self.material_score -= MATERIAL_SCORE[promotion_piece as usize];

                if captured_piece != NO_PIECE {
                    self.board[to_square] = captured_piece;
                    self.bitboards[captured_piece as usize] ^= to_square_mask;
                    self.network.add(captured_piece, to_square);
                    self.material_score += MATERIAL_SCORE[captured_piece as usize];
                }
            }
            ChessMoveType::Castle => {
                self.board[from_square] = moved_piece;
                self.board[to_square] = captured_piece;

                self.bitboards[moved_piece as usize] ^= from_square_mask | to_square_mask;

                self.network.add(moved_piece, from_square);
                self.network.remove(moved_piece, to_square);

                if moved_piece == WK {
                    self.white_king_square = from_square;

                    if to_square == G1 {
                        self.board[H1] = WR;
                        self.board[F1] = NO_PIECE;
                        self.bitboards[WR as usize] ^= SQUARE_MASKS[H1] | SQUARE_MASKS[F1];
                        self.network.add(WR, H1);
                        self.network.remove(WR, F1);
                    } else {
                        self.board[A1] = WR;
                        self.board[D1] = NO_PIECE;
                        self.bitboards[WR as usize] ^= SQUARE_MASKS[A1] | SQUARE_MASKS[D1];
                        self.network.add(WR, A1);
                        self.network.remove(WR, D1);
                    }
                } else if moved_piece == BK {
                    self.black_king_square = from_square;

                    if to_square == G8 {
                        self.board[H8] = BR;
                        self.board[F8] = NO_PIECE;
                        self.bitboards[BR as usize] ^= SQUARE_MASKS[H8] | SQUARE_MASKS[F8];
                        self.network.add(BR, H8);
                        self.network.remove(BR, F8);
                    } else {
                        self.board[A8] = BR;
                        self.board[D8] = NO_PIECE;
                        self.bitboards[BR as usize] ^= SQUARE_MASKS[A8] | SQUARE_MASKS[D8];
                        self.network.add(BR, A8);
                        self.network.remove(BR, D8);
                    }
                }
            }
            ChessMoveType::EnPassant => {
                self.board[from_square] = moved_piece;
                self.board[to_square] = NO_PIECE;

                self.bitboards[moved_piece as usize] ^= from_square_mask | to_square_mask;

                self.network.add(moved_piece, from_square);
                self.network.remove(moved_piece, to_square);

                if moved_piece == WP {
                    let captured_square = to_square - CHESS_FILE_COUNT;
                    self.board[captured_square] = BP;
                    self.bitboards[BP as usize] ^= SQUARE_MASKS[captured_square];
                    self.network.add(BP, captured_square);
                    self.material_score += MATERIAL_SCORE[BP as usize];
                } else {
                    let captured_square = to_square + CHESS_FILE_COUNT;
                    self.board[captured_square] = WP;
                    self.bitboards[WP as usize] ^= SQUARE_MASKS[captured_square];
                    self.network.add(WP, captured_square);
                    self.material_score += MATERIAL_SCORE[WP as usize];
                }
            }
            ChessMoveType::CreateEnPassant => {
                self.board[from_square] = moved_piece;
                self.board[to_square] = captured_piece;

                self.bitboards[moved_piece as usize] ^= from_square_mask | to_square_mask;

                self.network.add(moved_piece, from_square);
                self.network.remove(moved_piece, to_square);
            }
        }
    }

    #[inline(always)]
    fn modify_hash(&mut self) {
        self.hash_key ^= PLAYER_HASH[self.player as usize];
        self.hash_key ^= CASTLING_FLAG_HASH[self.castling_flag as usize];

        if self.enpassant_square != NO_SQUARE {
            self.hash_key ^= ENPASSANT_SQUARE_HASH[self.enpassant_square];
        }
    }
}
