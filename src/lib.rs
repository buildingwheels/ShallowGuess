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

#![feature(portable_simd)]

pub mod bit_masks;
pub mod chess_game;
pub mod chess_move_gen;
pub mod chess_position;
pub mod def;
pub mod engine_info;
pub mod fen;
pub mod generated;
pub mod network;
pub mod prng;
pub mod search_engine;
pub mod time;
pub mod transpos;
pub mod types;
pub mod uci;
pub mod util;
