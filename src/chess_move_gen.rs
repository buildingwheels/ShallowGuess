use crate::bit_masks::{
    get_bishop_attack_mask, get_rook_attack_mask, BLACK_PAWN_ATTACK_MASKS, EMPTY_MASK,
    KING_MOVE_MASKS, KNIGHT_MOVE_MASKS, SQUARE_MASKS, WHITE_PAWN_ATTACK_MASKS,
};
use crate::chess_position::ChessPosition;
use crate::def::{
    B1, B8, BB, BK, BLACK, BN, BP, BQ, BR, C1, C8, CASTLING_FLAG_BLACK_KING_SIDE,
    CASTLING_FLAG_BLACK_QUEEN_SIDE, CASTLING_FLAG_WHITE_KING_SIDE, CASTLING_FLAG_WHITE_QUEEN_SIDE,
    D1, D8, E1, E8, F1, F8, G1, G8, NO_PIECE, NO_SQUARE, RANK_2, RANK_7, UP_DELTA, WB, WHITE, WK,
    WN, WP, WQ, WR,
};
use crate::process_occupied_indices;
use crate::types::ChessMoveType::{Castle, CreateEnPassant, EnPassant, Promotion, Regular};
use crate::types::{ChessMove, ChessSquare, Player};
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

pub fn generate_captures_and_promotions(chess_position: &ChessPosition) -> Vec<ChessMove> {
    let mut chess_moves = Vec::new();

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

pub fn generate_quiet_moves(chess_position: &ChessPosition, in_check: bool) -> Vec<ChessMove> {
    let mut chess_moves = Vec::new();

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

pub fn is_invalid_position(chess_position: &ChessPosition) -> bool {
    is_in_check(chess_position, chess_position.player ^ BLACK)
}

pub fn is_in_check(chess_position: &ChessPosition, player: Player) -> bool {
    if player == WHITE {
        is_square_under_attack(chess_position.white_king_square, BLACK, chess_position)
    } else {
        is_square_under_attack(chess_position.black_king_square, WHITE, chess_position)
    }
}

fn is_square_under_attack(
    chess_square: ChessSquare,
    attack_player: Player,
    chess_position: &ChessPosition,
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
