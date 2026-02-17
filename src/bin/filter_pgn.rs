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

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, LineWriter, Write};
use std::time::Instant;

#[derive(Debug, Clone)]
enum FilterOp {
    Eq,
    Neq,
    Gt,
    Lt,
}

#[derive(Debug, Clone)]
struct Filter {
    tag: String,
    op: FilterOp,
    value: String,
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut input_file: Option<&str> = None;
    let mut output_file: Option<&str> = None;
    let mut filter_str: Option<&str> = None;
    let mut filter_if_missing_tag = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--filter-if-missing-tag" => {
                filter_if_missing_tag = true;
            }
            arg if arg.starts_with("--") => {
                eprintln!("Unknown argument: {}", arg);
                std::process::exit(1);
            }
            _ => {
                if input_file.is_none() {
                    input_file = Some(&args[i]);
                } else if output_file.is_none() {
                    output_file = Some(&args[i]);
                } else if filter_str.is_none() {
                    filter_str = Some(&args[i]);
                } else {
                    eprintln!(
                        "Usage: {} [options] <input.pgn> <output.pgn> <filters>",
                        args[0]
                    );
                    eprintln!("Options:");
                    eprintln!("  --filter-if-missing-tag    Filter out games with missing tags (default: false)");
                    std::process::exit(1);
                }
            }
        }
        i += 1;
    }

    let input_file = match input_file {
        Some(f) => f,
        None => {
            eprintln!(
                "Usage: {} [options] <input.pgn> <output.pgn> <filters>",
                args[0]
            );
            eprintln!("Options:");
            eprintln!(
                "  --filter-if-missing-tag    Filter out games with missing tags (default: false)"
            );
            std::process::exit(1);
        }
    };

    let output_file = match output_file {
        Some(f) => f,
        None => {
            eprintln!(
                "Usage: {} [options] <input.pgn> <output.pgn> <filters>",
                args[0]
            );
            eprintln!("Options:");
            eprintln!(
                "  --filter-if-missing-tag    Filter out games with missing tags (default: false)"
            );
            std::process::exit(1);
        }
    };

    let filter_str = match filter_str {
        Some(f) => f,
        None => {
            eprintln!(
                "Usage: {} [options] <input.pgn> <output.pgn> <filters>",
                args[0]
            );
            eprintln!("Options:");
            eprintln!(
                "  --filter-if-missing-tag    Filter out games with missing tags (default: false)"
            );
            std::process::exit(1);
        }
    };

    let filters = parse_filters(filter_str);

    let start_time = Instant::now();
    let (matched, total) = filter_pgn(input_file, output_file, &filters, filter_if_missing_tag);

    println!(
        "Filtered {} of {} games to {} in {:.2}s",
        matched,
        total,
        output_file,
        start_time.elapsed().as_secs_f64()
    );
}

fn parse_filters(filter_str: &str) -> Vec<Filter> {
    let mut filters = Vec::new();

    for part in filter_str.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        if let Some((tag, value)) = part.split_once("!=") {
            filters.push(Filter {
                tag: tag.trim().to_string(),
                op: FilterOp::Neq,
                value: value.trim().to_string(),
            });
        } else if let Some((tag, value)) = part.split_once('=') {
            filters.push(Filter {
                tag: tag.trim().to_string(),
                op: FilterOp::Eq,
                value: value.trim().to_string(),
            });
        } else if let Some((tag, value)) = part.split_once('>') {
            filters.push(Filter {
                tag: tag.trim().to_string(),
                op: FilterOp::Gt,
                value: value.trim().to_string(),
            });
        } else if let Some((tag, value)) = part.split_once('<') {
            filters.push(Filter {
                tag: tag.trim().to_string(),
                op: FilterOp::Lt,
                value: value.trim().to_string(),
            });
        }
    }

    filters
}

fn filter_pgn(
    input_path: &str,
    output_path: &str,
    filters: &[Filter],
    filter_if_missing_tag: bool,
) -> (usize, usize) {
    let file = File::open(input_path).unwrap_or_else(|e| {
        eprintln!("Error opening input file: {}", e);
        std::process::exit(1);
    });

    let output = File::create(output_path).unwrap_or_else(|e| {
        eprintln!("Error creating output file: {}", e);
        std::process::exit(1);
    });

    let reader = BufReader::new(file);
    let mut writer = LineWriter::new(output);

    let mut total_games = 0;
    let mut matched_games = 0;

    let mut current_game = String::new();
    let mut game_tags: HashMap<String, String> = HashMap::new();
    let mut in_moves = false;

    for line in reader.lines() {
        let line = line.unwrap_or_else(|e| {
            eprintln!("Error reading line: {}", e);
            String::new()
        });

        if line.starts_with('[') && line.ends_with(']') {
            if in_moves && !current_game.is_empty() {
                if matches_filters(&game_tags, filters, filter_if_missing_tag) {
                    writer.write_all(current_game.as_bytes()).unwrap();
                    matched_games += 1;
                }
                total_games += 1;
                current_game.clear();
                game_tags.clear();
            }

            in_moves = false;

            if let Some((tag, value)) = parse_tag_line(&line) {
                game_tags.insert(tag, value);
            }

            current_game.push_str(&line);
            current_game.push('\n');
        } else if line.is_empty() {
            if !in_moves && !current_game.is_empty() {
                current_game.push('\n');
                in_moves = true;
            } else if in_moves {
                current_game.push('\n');
            }
        } else {
            in_moves = true;
            current_game.push_str(&line);
            current_game.push('\n');
        }
    }

    if !current_game.is_empty() {
        if matches_filters(&game_tags, filters, filter_if_missing_tag) {
            writer.write_all(current_game.as_bytes()).unwrap();
            matched_games += 1;
        }
        total_games += 1;
    }

    writer.flush().unwrap();

    (matched_games, total_games)
}

fn parse_tag_line(line: &str) -> Option<(String, String)> {
    let line = line.trim_start_matches('[').trim_end_matches(']');

    let (tag, rest) = line.split_once(' ')?;
    let value = rest.trim_start_matches('"').trim_end_matches('"');

    Some((tag.to_string(), value.to_string()))
}

fn matches_filters(
    game_tags: &HashMap<String, String>,
    filters: &[Filter],
    filter_if_missing_tag: bool,
) -> bool {
    for filter in filters {
        let game_value = match game_tags.get(&filter.tag) {
            Some(v) => v,
            None => {
                if filter_if_missing_tag {
                    return false;
                } else {
                    continue;
                }
            }
        };

        let matches = match filter.op {
            FilterOp::Eq => game_value == &filter.value,
            FilterOp::Neq => game_value != &filter.value,
            FilterOp::Gt => match (game_value.parse::<i64>(), filter.value.parse::<i64>()) {
                (Ok(gv), Ok(fv)) => gv > fv,
                _ => false,
            },
            FilterOp::Lt => match (game_value.parse::<i64>(), filter.value.parse::<i64>()) {
                (Ok(gv), Ok(fv)) => gv < fv,
                _ => false,
            },
        };

        if !matches {
            return false;
        }
    }
    true
}
