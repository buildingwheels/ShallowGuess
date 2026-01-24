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

use crate::bit_masks::{
    get_bishop_attack_mask, get_rook_attack_mask, BLACK_PAWN_ATTACK_MASKS, EMPTY_MASK,
    KING_MOVE_MASKS, KNIGHT_MOVE_MASKS, SQUARE_MASKS, WHITE_PAWN_ATTACK_MASKS,
};
use crate::chess_position::ChessPosition;
use crate::def::{
    B1, B8, BB, BK, BLACK, BN, BP, BQ, BR, C1, C8, CASTLING_FLAG_BLACK_KING_SIDE,
    CASTLING_FLAG_BLACK_QUEEN_SIDE, CASTLING_FLAG_WHITE_KING_SIDE, CASTLING_FLAG_WHITE_QUEEN_SIDE,
    D1, D8, E1, E8, F1, F8, G1, G8, NO_PIECE, NO_SQUARE, PIECE_TYPE_COUNT, PIECE_VALS, RANK_2,
    RANK_7, UP_DELTA, WB, WHITE, WK, WN, WP, WQ, WR,
};
use crate::network::Network;
use crate::process_occupied_indices;
use crate::types::ChessMoveType::{Castle, CreateEnPassant, EnPassant, Promotion, Regular};
use crate::types::{BitBoard, ChessMove, ChessPiece, ChessSquare, Player, Score};
use crate::util::get_rank;

macro_rules! generate_white_promotions {
    ($from_square:expr, $to_square:expr, $chess_moves:expr) => {
        $chess_moves.push(ChessMove {
            from_square: $from_square,
            to_square: $to_square,
            promotion_piece: WQ,
            move_type: Promotion,
        });

        $chess_moves.push(ChessMove {
            from_square: $from_square,
            to_square: $to_square,
            promotion_piece: WR,
            move_type: Promotion,
        });

        $chess_moves.push(ChessMove {
            from_square: $from_square,
            to_square: $to_square,
            promotion_piece: WB,
            move_type: Promotion,
        });

        $chess_moves.push(ChessMove {
            from_square: $from_square,
            to_square: $to_square,
            promotion_piece: WN,
            move_type: Promotion,
        });
    };
}

macro_rules! generate_black_promotions {
    ($from_square:expr, $to_square:expr, $chess_moves:expr) => {
        $chess_moves.push(ChessMove {
            from_square: $from_square,
            to_square: $to_square,
            promotion_piece: BQ,
            move_type: Promotion,
        });

        $chess_moves.push(ChessMove {
            from_square: $from_square,
            to_square: $to_square,
            promotion_piece: BR,
            move_type: Promotion,
        });

        $chess_moves.push(ChessMove {
            from_square: $from_square,
            to_square: $to_square,
            promotion_piece: BB,
            move_type: Promotion,
        });

        $chess_moves.push(ChessMove {
            from_square: $from_square,
            to_square: $to_square,
            promotion_piece: BN,
            move_type: Promotion,
        });
    };
}

macro_rules! generate_moves_from_move_mask {
    ($from_square:expr, $move_mask:expr, $chess_moves:expr) => {
        process_occupied_indices!($move_mask, |next_to_square| {
            $chess_moves.push(ChessMove {
                from_square: $from_square,
                to_square: next_to_square,
                promotion_piece: NO_PIECE,
                move_type: Regular,
            });
        })
    };
}

macro_rules! generate_white_promotions_from_move_mask {
    ($from_square:expr, $move_mask:expr, $chess_moves:expr) => {
        process_occupied_indices!($move_mask, |next_to_square| {
            generate_white_promotions!($from_square, next_to_square, $chess_moves);
        })
    };
}

macro_rules! generate_black_promotions_from_move_mask {
    ($from_square:expr, $move_mask:expr, $chess_moves:expr) => {
        process_occupied_indices!($move_mask, |next_to_square| {
            generate_black_promotions!($from_square, next_to_square, $chess_moves);
        })
    };
}

const MAX_POSSIBLE_MOVES_COUNT: usize = 256;

pub fn generate_captures_and_promotions<N: Network>(
    chess_position: &ChessPosition<N>,
) -> Vec<ChessMove> {
    let mut chess_moves = Vec::with_capacity(MAX_POSSIBLE_MOVES_COUNT);

    let white_mask = chess_position.white_all_bitboard;
    let black_mask = chess_position.black_all_bitboard;
    let occupy_mask = white_mask | black_mask;

    if chess_position.player == WHITE {
        process_occupied_indices!(white_mask, |piece_index| {
            match chess_position.board[piece_index] {
                WP => {
                    let rank = get_rank(piece_index);
                    if rank == RANK_7 {
                        generate_white_promotions_from_move_mask!(
                            piece_index,
                            WHITE_PAWN_ATTACK_MASKS[piece_index] & black_mask,
                            &mut chess_moves
                        );

                        if chess_position.board[piece_index + UP_DELTA] == NO_PIECE {
                            generate_white_promotions!(
                                piece_index,
                                piece_index + UP_DELTA,
                                &mut chess_moves
                            );
                        }
                    } else {
                        let pawn_attack_mask = WHITE_PAWN_ATTACK_MASKS[piece_index];
                        let enpassant_square_mask = if chess_position.enpassant_square == NO_SQUARE
                        {
                            EMPTY_MASK
                        } else {
                            SQUARE_MASKS[chess_position.enpassant_square]
                        };

                        if rank != RANK_2 && pawn_attack_mask & enpassant_square_mask != 0 {
                            chess_moves.push(ChessMove {
                                from_square: piece_index,
                                to_square: chess_position.enpassant_square,
                                promotion_piece: NO_PIECE,
                                move_type: EnPassant,
                            });
                        }

                        generate_moves_from_move_mask!(
                            piece_index,
                            pawn_attack_mask & black_mask,
                            &mut chess_moves
                        );
                    }
                }
                WN => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        KNIGHT_MOVE_MASKS[piece_index] & black_mask,
                        &mut chess_moves
                    );
                }
                WB => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        get_bishop_attack_mask(occupy_mask, piece_index) & black_mask,
                        &mut chess_moves
                    );
                }
                WR => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        get_rook_attack_mask(occupy_mask, piece_index) & black_mask,
                        &mut chess_moves
                    );
                }
                WQ => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        (get_bishop_attack_mask(occupy_mask, piece_index)
                            | get_rook_attack_mask(occupy_mask, piece_index))
                            & black_mask,
                        &mut chess_moves
                    );
                }
                WK => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        KING_MOVE_MASKS[piece_index] & black_mask,
                        &mut chess_moves
                    );
                }
                _ => {}
            }
        });
    } else {
        process_occupied_indices!(black_mask, |piece_index| {
            match chess_position.board[piece_index] {
                BP => {
                    let rank = get_rank(piece_index);
                    if rank == RANK_2 {
                        generate_black_promotions_from_move_mask!(
                            piece_index,
                            BLACK_PAWN_ATTACK_MASKS[piece_index] & white_mask,
                            &mut chess_moves
                        );

                        if chess_position.board[piece_index - UP_DELTA] == NO_PIECE {
                            generate_black_promotions!(
                                piece_index,
                                piece_index - UP_DELTA,
                                &mut chess_moves
                            );
                        }
                    } else {
                        let pawn_attack_mask = BLACK_PAWN_ATTACK_MASKS[piece_index];
                        let enpassant_square_mask = if chess_position.enpassant_square == NO_SQUARE
                        {
                            EMPTY_MASK
                        } else {
                            SQUARE_MASKS[chess_position.enpassant_square]
                        };

                        if rank != RANK_7 && pawn_attack_mask & enpassant_square_mask != 0 {
                            chess_moves.push(ChessMove {
                                from_square: piece_index,
                                to_square: chess_position.enpassant_square,
                                promotion_piece: NO_PIECE,
                                move_type: EnPassant,
                            });
                        }

                        generate_moves_from_move_mask!(
                            piece_index,
                            pawn_attack_mask & white_mask,
                            &mut chess_moves
                        );
                    }
                }
                BN => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        KNIGHT_MOVE_MASKS[piece_index] & white_mask,
                        &mut chess_moves
                    );
                }
                BB => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        get_bishop_attack_mask(occupy_mask, piece_index) & white_mask,
                        &mut chess_moves
                    );
                }
                BR => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        get_rook_attack_mask(occupy_mask, piece_index) & white_mask,
                        &mut chess_moves
                    );
                }
                BQ => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        (get_bishop_attack_mask(occupy_mask, piece_index)
                            | get_rook_attack_mask(occupy_mask, piece_index))
                            & white_mask,
                        &mut chess_moves
                    );
                }
                BK => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        KING_MOVE_MASKS[piece_index] & white_mask,
                        &mut chess_moves
                    );
                }
                _ => {}
            }
        });
    }

    chess_moves
}

pub fn generate_quiet_moves<N: Network>(
    chess_position: &ChessPosition<N>,
    in_check: bool,
) -> Vec<ChessMove> {
    let mut chess_moves = Vec::with_capacity(MAX_POSSIBLE_MOVES_COUNT);

    let white_mask = chess_position.white_all_bitboard;
    let black_mask = chess_position.black_all_bitboard;
    let occupy_mask = white_mask | black_mask;
    let non_occupy_mask = !occupy_mask;

    if chess_position.player == WHITE {
        process_occupied_indices!(white_mask, |piece_index| {
            match chess_position.board[piece_index] {
                WP => {
                    let rank = get_rank(piece_index);

                    if rank != RANK_7 {
                        if chess_position.board[piece_index + UP_DELTA] == NO_PIECE {
                            chess_moves.push(ChessMove {
                                from_square: piece_index,
                                to_square: piece_index + UP_DELTA,
                                promotion_piece: NO_PIECE,
                                move_type: Regular,
                            });

                            if rank == RANK_2
                                && chess_position.board[piece_index + UP_DELTA + UP_DELTA]
                                    == NO_PIECE
                            {
                                chess_moves.push(ChessMove {
                                    from_square: piece_index,
                                    to_square: piece_index + UP_DELTA + UP_DELTA,
                                    promotion_piece: NO_PIECE,
                                    move_type: CreateEnPassant,
                                });
                            }
                        }
                    }
                }
                WN => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        KNIGHT_MOVE_MASKS[piece_index] & non_occupy_mask,
                        &mut chess_moves
                    );
                }
                WB => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        get_bishop_attack_mask(occupy_mask, piece_index) & non_occupy_mask,
                        &mut chess_moves
                    );
                }
                WR => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        get_rook_attack_mask(occupy_mask, piece_index) & non_occupy_mask,
                        &mut chess_moves
                    );
                }
                WQ => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        (get_bishop_attack_mask(occupy_mask, piece_index)
                            | get_rook_attack_mask(occupy_mask, piece_index))
                            & non_occupy_mask,
                        &mut chess_moves
                    );
                }
                WK => {
                    if !in_check {
                        if chess_position.castling_flag & CASTLING_FLAG_WHITE_KING_SIDE != 0
                            && chess_position.board[F1] == NO_PIECE
                            && chess_position.board[G1] == NO_PIECE
                            && !is_square_under_attack(F1, BLACK, chess_position)
                            && !is_square_under_attack(G1, BLACK, chess_position)
                        {
                            chess_moves.push(ChessMove {
                                from_square: E1,
                                to_square: G1,
                                promotion_piece: NO_PIECE,
                                move_type: Castle,
                            });
                        }

                        if chess_position.castling_flag & CASTLING_FLAG_WHITE_QUEEN_SIDE != 0
                            && chess_position.board[D1] == NO_PIECE
                            && chess_position.board[C1] == NO_PIECE
                            && chess_position.board[B1] == NO_PIECE
                            && !is_square_under_attack(D1, BLACK, chess_position)
                            && !is_square_under_attack(C1, BLACK, chess_position)
                        {
                            chess_moves.push(ChessMove {
                                from_square: E1,
                                to_square: C1,
                                promotion_piece: NO_PIECE,
                                move_type: Castle,
                            });
                        }
                    }

                    generate_moves_from_move_mask!(
                        piece_index,
                        KING_MOVE_MASKS[piece_index] & non_occupy_mask,
                        &mut chess_moves
                    );
                }
                _ => {}
            }
        });
    } else {
        process_occupied_indices!(black_mask, |piece_index| {
            match chess_position.board[piece_index] {
                BP => {
                    let rank = get_rank(piece_index);

                    if rank != RANK_2 {
                        if chess_position.board[piece_index - UP_DELTA] == NO_PIECE {
                            chess_moves.push(ChessMove {
                                from_square: piece_index,
                                to_square: piece_index - UP_DELTA,
                                promotion_piece: NO_PIECE,
                                move_type: Regular,
                            });

                            if rank == RANK_7
                                && chess_position.board[piece_index - UP_DELTA - UP_DELTA]
                                    == NO_PIECE
                            {
                                chess_moves.push(ChessMove {
                                    from_square: piece_index,
                                    to_square: piece_index - UP_DELTA - UP_DELTA,
                                    promotion_piece: NO_PIECE,
                                    move_type: CreateEnPassant,
                                });
                            }
                        }
                    }
                }
                BN => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        KNIGHT_MOVE_MASKS[piece_index] & non_occupy_mask,
                        &mut chess_moves
                    );
                }
                BB => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        get_bishop_attack_mask(occupy_mask, piece_index) & non_occupy_mask,
                        &mut chess_moves
                    );
                }
                BR => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        get_rook_attack_mask(occupy_mask, piece_index) & non_occupy_mask,
                        &mut chess_moves
                    );
                }
                BQ => {
                    generate_moves_from_move_mask!(
                        piece_index,
                        (get_bishop_attack_mask(occupy_mask, piece_index)
                            | get_rook_attack_mask(occupy_mask, piece_index))
                            & non_occupy_mask,
                        &mut chess_moves
                    );
                }
                BK => {
                    if !in_check {
                        if chess_position.castling_flag & CASTLING_FLAG_BLACK_KING_SIDE != 0
                            && chess_position.board[F8] == NO_PIECE
                            && chess_position.board[G8] == NO_PIECE
                            && !is_square_under_attack(F8, WHITE, chess_position)
                            && !is_square_under_attack(G8, WHITE, chess_position)
                        {
                            chess_moves.push(ChessMove {
                                from_square: E8,
                                to_square: G8,
                                promotion_piece: NO_PIECE,
                                move_type: Castle,
                            });
                        }

                        if chess_position.castling_flag & CASTLING_FLAG_BLACK_QUEEN_SIDE != 0
                            && chess_position.board[D8] == NO_PIECE
                            && chess_position.board[C8] == NO_PIECE
                            && chess_position.board[B8] == NO_PIECE
                            && !is_square_under_attack(D8, WHITE, chess_position)
                            && !is_square_under_attack(C8, WHITE, chess_position)
                        {
                            chess_moves.push(ChessMove {
                                from_square: E8,
                                to_square: C8,
                                promotion_piece: NO_PIECE,
                                move_type: Castle,
                            });
                        }
                    }

                    generate_moves_from_move_mask!(
                        piece_index,
                        KING_MOVE_MASKS[piece_index] & non_occupy_mask,
                        &mut chess_moves
                    );
                }
                _ => {}
            }
        });
    }

    chess_moves
}

pub fn is_invalid_position<N: Network>(chess_position: &ChessPosition<N>) -> bool {
    is_player_in_check(chess_position, chess_position.player ^ BLACK)
}

pub fn is_in_check<N: Network>(chess_position: &ChessPosition<N>) -> bool {
    if chess_position.player == WHITE {
        is_square_under_attack(chess_position.white_king_square, BLACK, chess_position)
    } else {
        is_square_under_attack(chess_position.black_king_square, WHITE, chess_position)
    }
}

pub fn is_player_in_check<N: Network>(chess_position: &ChessPosition<N>, player: Player) -> bool {
    if player == WHITE {
        is_square_under_attack(chess_position.white_king_square, BLACK, chess_position)
    } else {
        is_square_under_attack(chess_position.black_king_square, WHITE, chess_position)
    }
}

pub fn static_exchange_evaluation<N: Network>(
    to_square: ChessSquare,
    chess_position: &mut ChessPosition<N>,
) -> Score {
    let bitboards = &mut chess_position.bitboards;
    let occupy_mask = chess_position.white_all_bitboard | chess_position.black_all_bitboard;

    let mut attackers_white =
        get_all_attackers_for_player(to_square, WHITE, bitboards, occupy_mask);
    attackers_white.sort_by(|&(first_chess_piece, _), &(second_chess_piece, _)| {
        PIECE_VALS[first_chess_piece as usize].cmp(&PIECE_VALS[second_chess_piece as usize])
    });

    let mut attackers_black =
        get_all_attackers_for_player(to_square, BLACK, bitboards, occupy_mask);
    attackers_black.sort_by(|&(first_chess_piece, _), &(second_chess_piece, _)| {
        PIECE_VALS[first_chess_piece as usize].cmp(&PIECE_VALS[second_chess_piece as usize])
    });

    let mut known_attackers_mask = EMPTY_MASK;

    for &(_, attacker_square) in &attackers_white {
        known_attackers_mask |= SQUARE_MASKS[attacker_square];
    }

    for &(_, attacker_square) in &attackers_black {
        known_attackers_mask |= SQUARE_MASKS[attacker_square];
    }

    static_exchange_evaluation_recursive(
        to_square,
        chess_position.player,
        chess_position.board[to_square],
        &mut attackers_white,
        &mut attackers_black,
        known_attackers_mask,
        bitboards,
        occupy_mask,
    )
    .max(0)
}

fn static_exchange_evaluation_recursive(
    to_square: ChessSquare,
    current_player: Player,
    previous_attacker: ChessPiece,
    attackers_white: &mut Vec<(ChessPiece, ChessSquare)>,
    attackers_black: &mut Vec<(ChessPiece, ChessSquare)>,
    mut known_attackers_mask: BitBoard,
    bitboards: &mut [BitBoard; PIECE_TYPE_COUNT],
    occupy_mask: BitBoard,
) -> Score {
    let (next_attacker, next_attacker_square) = if current_player == WHITE {
        if attackers_white.is_empty() {
            return 0;
        }

        attackers_white.remove(0)
    } else {
        if attackers_black.is_empty() {
            return 0;
        }

        attackers_black.remove(0)
    };

    let next_attacker_square_mask = SQUARE_MASKS[next_attacker_square];

    bitboards[next_attacker as usize] &= !next_attacker_square_mask;

    let updated_occupy_mask = occupy_mask & !next_attacker_square_mask;

    let new_attackers_white = get_ray_attackers_for_player(
        to_square,
        WHITE,
        bitboards,
        updated_occupy_mask,
        known_attackers_mask,
    );

    if !new_attackers_white.is_empty() {
        for (_, attacker_square) in &new_attackers_white {
            known_attackers_mask |= SQUARE_MASKS[*attacker_square];
        }

        attackers_white.extend(new_attackers_white);
        attackers_white.sort_by(|&(first_chess_piece, _), &(second_chess_piece, _)| {
            PIECE_VALS[first_chess_piece as usize].cmp(&PIECE_VALS[second_chess_piece as usize])
        });
    }

    let new_attackers_black = get_ray_attackers_for_player(
        to_square,
        BLACK,
        bitboards,
        updated_occupy_mask,
        known_attackers_mask,
    );

    if !new_attackers_black.is_empty() {
        for (_, attacker_square) in &new_attackers_black {
            known_attackers_mask |= SQUARE_MASKS[*attacker_square];
        }

        attackers_black.extend(new_attackers_black);
        attackers_black.sort_by(|&(first_chess_piece, _), &(second_chess_piece, _)| {
            PIECE_VALS[first_chess_piece as usize].cmp(&PIECE_VALS[second_chess_piece as usize])
        });
    }

    let score = PIECE_VALS[previous_attacker as usize]
        - static_exchange_evaluation_recursive(
            to_square,
            current_player ^ BLACK,
            next_attacker,
            attackers_white,
            attackers_black,
            known_attackers_mask,
            bitboards,
            updated_occupy_mask,
        );

    bitboards[next_attacker as usize] |= next_attacker_square_mask;

    score.max(0)
}

fn get_all_attackers_for_player(
    chess_square: ChessSquare,
    attack_player: Player,
    bitboards: &[BitBoard; PIECE_TYPE_COUNT],
    occupy_mask: BitBoard,
) -> Vec<(ChessPiece, ChessSquare)> {
    let mut attackers = Vec::new();

    if attack_player == BLACK {
        process_occupied_indices!(
            WHITE_PAWN_ATTACK_MASKS[chess_square] & bitboards[BP as usize],
            |attacker_square| {
                attackers.push((BP, attacker_square));
            }
        );

        process_occupied_indices!(
            KNIGHT_MOVE_MASKS[chess_square] & bitboards[BN as usize],
            |attacker_square| {
                attackers.push((BN, attacker_square));
            }
        );

        process_occupied_indices!(
            KING_MOVE_MASKS[chess_square] & bitboards[BK as usize],
            |attacker_square| {
                attackers.push((BK, attacker_square));
            }
        );

        let bishop_attackers_mask = get_bishop_attack_mask(occupy_mask, chess_square);

        process_occupied_indices!(
            bishop_attackers_mask & bitboards[BB as usize],
            |attacker_square| {
                attackers.push((BB, attacker_square));
            }
        );

        process_occupied_indices!(
            bishop_attackers_mask & bitboards[BQ as usize],
            |attacker_square| {
                attackers.push((BQ, attacker_square));
            }
        );

        let rook_attackers_mask = get_rook_attack_mask(occupy_mask, chess_square);

        process_occupied_indices!(
            rook_attackers_mask & bitboards[BR as usize],
            |attacker_square| {
                attackers.push((BR, attacker_square));
            }
        );

        process_occupied_indices!(
            rook_attackers_mask & bitboards[BQ as usize],
            |attacker_square| {
                attackers.push((BQ, attacker_square));
            }
        );
    } else {
        process_occupied_indices!(
            BLACK_PAWN_ATTACK_MASKS[chess_square] & bitboards[WP as usize],
            |attacker_square| {
                attackers.push((WP, attacker_square));
            }
        );

        process_occupied_indices!(
            KNIGHT_MOVE_MASKS[chess_square] & bitboards[WN as usize],
            |attacker_square| {
                attackers.push((WN, attacker_square));
            }
        );

        process_occupied_indices!(
            KING_MOVE_MASKS[chess_square] & bitboards[WK as usize],
            |attacker_square| {
                attackers.push((WK, attacker_square));
            }
        );

        let bishop_attackers_mask = get_bishop_attack_mask(occupy_mask, chess_square);

        process_occupied_indices!(
            bishop_attackers_mask & bitboards[WB as usize],
            |attacker_square| {
                attackers.push((WB, attacker_square));
            }
        );

        process_occupied_indices!(
            bishop_attackers_mask & bitboards[WQ as usize],
            |attacker_square| {
                attackers.push((WQ, attacker_square));
            }
        );

        let rook_attackers_mask = get_rook_attack_mask(occupy_mask, chess_square);

        process_occupied_indices!(
            rook_attackers_mask & bitboards[WR as usize],
            |attacker_square| {
                attackers.push((WR, attacker_square));
            }
        );

        process_occupied_indices!(
            rook_attackers_mask & bitboards[WQ as usize],
            |attacker_square| {
                attackers.push((WQ, attacker_square));
            }
        );
    }

    attackers
}

fn get_ray_attackers_for_player(
    chess_square: ChessSquare,
    attack_player: Player,
    bitboards: &[BitBoard; PIECE_TYPE_COUNT],
    occupy_mask: BitBoard,
    known_attackers_mask: BitBoard,
) -> Vec<(ChessPiece, ChessSquare)> {
    let mut attackers = Vec::new();

    if attack_player == BLACK {
        let bishop_attackers_mask = get_bishop_attack_mask(occupy_mask, chess_square);

        process_occupied_indices!(
            bishop_attackers_mask & bitboards[BB as usize],
            |attacker_square| {
                if known_attackers_mask & SQUARE_MASKS[attacker_square] == 0 {
                    attackers.push((BB, attacker_square));
                }
            }
        );

        process_occupied_indices!(
            bishop_attackers_mask & bitboards[BQ as usize],
            |attacker_square| {
                if known_attackers_mask & SQUARE_MASKS[attacker_square] == 0 {
                    attackers.push((BQ, attacker_square));
                }
            }
        );

        let rook_attackers_mask = get_rook_attack_mask(occupy_mask, chess_square);

        process_occupied_indices!(
            rook_attackers_mask & bitboards[BR as usize],
            |attacker_square| {
                if known_attackers_mask & SQUARE_MASKS[attacker_square] == 0 {
                    attackers.push((BR, attacker_square));
                }
            }
        );

        process_occupied_indices!(
            rook_attackers_mask & bitboards[BQ as usize],
            |attacker_square| {
                if known_attackers_mask & SQUARE_MASKS[attacker_square] == 0 {
                    attackers.push((BQ, attacker_square));
                }
            }
        );
    } else {
        let bishop_attackers_mask = get_bishop_attack_mask(occupy_mask, chess_square);

        process_occupied_indices!(
            bishop_attackers_mask & bitboards[WB as usize],
            |attacker_square| {
                if known_attackers_mask & SQUARE_MASKS[attacker_square] == 0 {
                    attackers.push((WB, attacker_square));
                }
            }
        );

        process_occupied_indices!(
            bishop_attackers_mask & bitboards[WQ as usize],
            |attacker_square| {
                if known_attackers_mask & SQUARE_MASKS[attacker_square] == 0 {
                    attackers.push((WQ, attacker_square));
                }
            }
        );

        let rook_attackers_mask = get_rook_attack_mask(occupy_mask, chess_square);

        process_occupied_indices!(
            rook_attackers_mask & bitboards[WR as usize],
            |attacker_square| {
                if known_attackers_mask & SQUARE_MASKS[attacker_square] == 0 {
                    attackers.push((WR, attacker_square));
                }
            }
        );

        process_occupied_indices!(
            rook_attackers_mask & bitboards[WQ as usize],
            |attacker_square| {
                if known_attackers_mask & SQUARE_MASKS[attacker_square] == 0 {
                    attackers.push((WQ, attacker_square));
                }
            }
        );
    }

    attackers
}

fn is_square_under_attack<N: Network>(
    chess_square: ChessSquare,
    attack_player: Player,
    chess_position: &ChessPosition<N>,
) -> bool {
    let bitboards = &chess_position.bitboards;
    let occupy_mask = chess_position.white_all_bitboard | chess_position.black_all_bitboard;

    if attack_player == BLACK {
        if WHITE_PAWN_ATTACK_MASKS[chess_square] & bitboards[BP as usize] != 0 {
            return true;
        }

        if KNIGHT_MOVE_MASKS[chess_square] & bitboards[BN as usize] != 0 {
            return true;
        }

        if KING_MOVE_MASKS[chess_square] & bitboards[BK as usize] != 0 {
            return true;
        }

        if get_bishop_attack_mask(occupy_mask, chess_square)
            & (bitboards[BB as usize] | bitboards[BQ as usize])
            != 0
        {
            return true;
        }

        if get_rook_attack_mask(occupy_mask, chess_square)
            & (bitboards[BR as usize] | bitboards[BQ as usize])
            != 0
        {
            return true;
        }
    } else {
        if BLACK_PAWN_ATTACK_MASKS[chess_square] & bitboards[WP as usize] != 0 {
            return true;
        }

        if KNIGHT_MOVE_MASKS[chess_square] & bitboards[WN as usize] != 0 {
            return true;
        }

        if KING_MOVE_MASKS[chess_square] & bitboards[WK as usize] != 0 {
            return true;
        }

        if get_bishop_attack_mask(occupy_mask, chess_square)
            & (bitboards[WB as usize] | bitboards[WQ as usize])
            != 0
        {
            return true;
        }

        if get_rook_attack_mask(occupy_mask, chess_square)
            & (bitboards[WR as usize] | bitboards[WQ as usize])
            != 0
        {
            return true;
        }
    }

    false
}
