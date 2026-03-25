use crate::def::{BUCKET_COUNT, PIECE_TYPE_COUNT};
use crate::types::Score;

pub const PIECE_VALUES: [[Score; PIECE_TYPE_COUNT]; BUCKET_COUNT] = [
    [    0,   184,   390,   413,   695,  1227,     0,  -199,  -394,  -412,  -694, -1207,     0],
    [    0,   168,   426,   455,   687,  1299,     0,  -183,  -430,  -461,  -686, -1272,     0],
    [    0,   100,   275,   297,   439,   691,     0,  -112,  -282,  -303,  -444,  -639,     0]
];
