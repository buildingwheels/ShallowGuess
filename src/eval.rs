use crate::{def::PIECE_TYPE_COUNT, types::Score};

pub const MATERIAL_SCORE: [Score; PIECE_TYPE_COUNT] = [
    0, 100, 300, 300, 500, 1000, 0, -100, -300, -300, -500, -1000, 0,
];
