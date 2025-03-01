use crate::def::{CHESS_SQUARE_COUNT, PIECE_TYPE_COUNT};

pub type ChessSquare = usize;
pub type ChessFile = usize;
pub type ChessRank = usize;
pub type ChessPiece = u8;
pub type BitBoard = u64;
pub type Player = u8;
pub type CastlingFlag = u8;
pub type ChessMoveCount = usize;
pub type NodeCount = usize;
pub type SearchDepth = u16;
pub type SearchPly = usize;
pub type MilliSeconds = u64;
pub type MegaBytes = usize;
pub type Score = i32;
pub type HashKey = usize;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum ChessMoveType {
    Regular,
    Castle,
    EnPassant,
    CreateEnPassant,
    Promotion,
}

#[derive(Clone, Copy, Debug)]
pub struct ChessMove {
    pub from_square: ChessSquare,
    pub to_square: ChessSquare,
    pub promotion_piece: ChessPiece,
    pub move_type: ChessMoveType,
}

impl ChessMove {
    pub fn is_empty(&self) -> bool {
        self.from_square == self.to_square
    }
}

impl PartialEq<ChessMove> for ChessMove {
    fn eq(&self, other: &ChessMove) -> bool {
        self.from_square == other.from_square && self.to_square == other.to_square
    }
}

pub const EMPTY_CHESS_MOVE: ChessMove = ChessMove {
    from_square: 0,
    to_square: 0,
    promotion_piece: 0,
    move_type: ChessMoveType::Regular,
};

pub struct SortableChessMove {
    pub chess_move: ChessMove,
    pub sort_score: Score,
}

impl PartialEq for SortableChessMove {
    fn eq(&self, other: &Self) -> bool {
        self.sort_score == other.sort_score
    }
}

impl Eq for SortableChessMove {}

impl PartialOrd for SortableChessMove {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.sort_score.partial_cmp(&other.sort_score)
    }
}

impl Ord for SortableChessMove {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort_score.cmp(&other.sort_score)
    }
}

pub type ChessSquareIndexToStrMap = [&'static str; CHESS_SQUARE_COUNT];
pub type ChessPieceToChessPieceCharMap = [char; PIECE_TYPE_COUNT];
