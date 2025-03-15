use crate::types::{ChessMoveCount, MilliSeconds};

pub struct TimeInfo {
    pub remaining_time_millis: MilliSeconds,
    pub increment_time_millis: MilliSeconds,
    pub remaining_move_count: ChessMoveCount,
}

impl TimeInfo {
    pub fn new() -> Self {
        TimeInfo {
            remaining_time_millis: 0,
            increment_time_millis: 0,
            remaining_move_count: 0,
        }
    }
}

const DEFAULT_REMAINING_MOVE_COUNT: ChessMoveCount = 40;
const INCREMENT_TIME_BUFFER_RATIO: MilliSeconds = 2;

pub fn calculate_optimal_time_for_next_move(
    time_control: &TimeInfo,
    is_first_move: bool,
) -> MilliSeconds {
    let remaining_move_count = if time_control.remaining_move_count == 0 {
        DEFAULT_REMAINING_MOVE_COUNT
    } else {
        time_control.remaining_move_count
    };

    let base_time = time_control.remaining_time_millis / remaining_move_count as MilliSeconds;

    let first_move_buffer = if is_first_move && remaining_move_count > 1 {
        base_time
    } else {
        0
    };

    base_time + time_control.increment_time_millis / INCREMENT_TIME_BUFFER_RATIO + first_move_buffer
}
