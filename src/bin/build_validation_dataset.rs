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

use shallow_guess::prng::RandGenerator;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::time::{SystemTime, UNIX_EPOCH};

const DRAW_RESULT: f64 = 0.5;

fn build_validation_dataset(
    input_file: &str,
    output_file: &str,
    num_samples: usize,
    draw_ratio: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);

    let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

    let mut draw_lines: Vec<String> = Vec::new();
    let mut non_draw_lines: Vec<String> = Vec::new();

    for line in &lines {
        if let Some(result_str) = line.split(',').nth(1) {
            if let Ok(result) = result_str.trim().parse::<f64>() {
                if (result - DRAW_RESULT).abs() < 1e-9 {
                    draw_lines.push(line.clone());
                } else {
                    non_draw_lines.push(line.clone());
                }
            }
        }
    }

    let target_draw_count = (num_samples as f64 * draw_ratio).round() as usize;
    let target_non_draw_count = num_samples - target_draw_count;

    if draw_lines.len() < target_draw_count {
        eprintln!(
            "Warning: Requested {} draw samples, but file only has {} draw lines",
            target_draw_count,
            draw_lines.len()
        );
    }
    if non_draw_lines.len() < target_non_draw_count {
        eprintln!(
            "Warning: Requested {} non-draw samples, but file only has {} non-draw lines",
            target_non_draw_count,
            non_draw_lines.len()
        );
    }

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let mut rng = RandGenerator::new(seed);

    let draw_count = target_draw_count.min(draw_lines.len());
    let non_draw_count = target_non_draw_count.min(non_draw_lines.len());
    let actual_samples = draw_count + non_draw_count;

    let mut selected_draw_indices = std::collections::HashSet::new();
    while selected_draw_indices.len() < draw_count {
        let index = (rng.next_f64() * draw_lines.len() as f64).floor() as usize;
        selected_draw_indices.insert(index);
    }

    let mut selected_non_draw_indices = std::collections::HashSet::new();
    while selected_non_draw_indices.len() < non_draw_count {
        let index = (rng.next_f64() * non_draw_lines.len() as f64).floor() as usize;
        selected_non_draw_indices.insert(index);
    }

    let mut output = File::create(output_file)?;

    for index in selected_draw_indices {
        writeln!(output, "{}", draw_lines[index])?;
    }

    for index in selected_non_draw_indices {
        writeln!(output, "{}", non_draw_lines[index])?;
    }

    eprintln!(
        "Selected {} samples ({} draws [{:.1}%], {} non-draws) from {} to {}",
        actual_samples,
        draw_count,
        draw_ratio * 100.0,
        non_draw_count,
        input_file,
        output_file,
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 5 {
        eprintln!(
            "Usage: {} <input_file> <output_file> <sample_count> <draw_ratio>",
            args[0]
        );
        eprintln!("  draw_ratio: fraction of samples that should be draws (e.g., 0.3 for 30%)");
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];
    let num_samples: usize = args[3]
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid number of samples"))?;
    let draw_ratio: f64 = args[4]
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid draw ratio"))?;

    if draw_ratio < 0.0 || draw_ratio > 1.0 {
        eprintln!("Error: draw_ratio must be between 0.0 and 1.0");
        std::process::exit(1);
    }

    match build_validation_dataset(input_file, output_file, num_samples, draw_ratio) {
        Ok(()) => Ok(()),
        Err(e) => {
            if let Some(io_err) = e.downcast_ref::<io::Error>() {
                match io_err.kind() {
                    io::ErrorKind::NotFound => {
                        eprintln!("Error: Input file '{}' not found.", input_file);
                    }
                    _ => {
                        eprintln!("Error: {}", e);
                    }
                }
            } else {
                eprintln!("Error: {}", e);
            }
            std::process::exit(1);
        }
    }
}
