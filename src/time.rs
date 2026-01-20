// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

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
const MIN_REMAINING_MOVE_COUNT_TO_USE_EXTRA_TIME: ChessMoveCount = 3;
const MIN_REMAINING_TIME_MILLIS_TO_USE_EXTRA_TIME: MilliSeconds = 10_000;
const BUFFER_TIME_MILLIS: MilliSeconds = 10;

pub fn calculate_optimal_time_for_next_move(
    time_control: &TimeInfo,
) -> (MilliSeconds, MilliSeconds) {
    let remaining_move_count = if time_control.remaining_move_count == 0 {
        DEFAULT_REMAINING_MOVE_COUNT
    } else {
        time_control.remaining_move_count
    };

    let base_time = time_control.remaining_time_millis / remaining_move_count as MilliSeconds;

    let extra_time = if remaining_move_count >= MIN_REMAINING_MOVE_COUNT_TO_USE_EXTRA_TIME
        && time_control.remaining_time_millis >= MIN_REMAINING_TIME_MILLIS_TO_USE_EXTRA_TIME
    {
        base_time
    } else {
        0
    };

    let mut total_base_time = base_time + time_control.increment_time_millis;

    if total_base_time > BUFFER_TIME_MILLIS {
        total_base_time -= BUFFER_TIME_MILLIS;
    }

    (total_base_time, extra_time)
}
