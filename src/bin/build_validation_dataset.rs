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

fn build_validation_dataset(
    input_file: &str,
    output_file: &str,
    num_samples: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);

    let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

    if num_samples > lines.len() {
        eprintln!(
            "Warning: Requested {} samples, but file only has {} lines",
            num_samples,
            lines.len()
        );
        eprintln!("Using all {} lines", lines.len());
    }

    let actual_samples = num_samples.min(lines.len());

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let mut rng = RandGenerator::new(seed);

    let mut selected_indices = std::collections::HashSet::new();
    while selected_indices.len() < actual_samples {
        let index = (rng.next_f64() * lines.len() as f64).floor() as usize;
        selected_indices.insert(index);
    }

    let mut output = File::create(output_file)?;

    for index in selected_indices {
        writeln!(output, "{}", lines[index])?;
    }

    eprintln!(
        "Selected {} random samples from {} to {}",
        actual_samples, input_file, output_file,
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!(
            "Usage: {} <input_file> <output_file> <sample_count>",
            args[0]
        );
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];
    let num_samples: usize = args[3]
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid number of samples"))?;

    match build_validation_dataset(input_file, output_file, num_samples) {
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
