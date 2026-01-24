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

use std::env;
use std::fs::{remove_file, File};
use std::io::{self, BufRead, BufReader, Error, ErrorKind, Write};

fn split_large_file(
    input_file: &str,
    lines_per_file: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);

    let mut file_number = 1;
    let mut line_count = 0;
    let mut output_file: Option<File> = None;

    for line in reader.lines() {
        let line = line?;

        if line_count % lines_per_file == 0 {
            if let Some(mut file) = output_file.take() {
                file.flush()?;
            }

            let output_filename = format!("{}_part_{}", input_file, file_number);
            output_file = Some(File::create(&output_filename)?);
            file_number += 1;
        }

        if let Some(ref mut file) = output_file {
            writeln!(file, "{}", line)?;
        }

        line_count += 1;
    }

    if let Some(mut file) = output_file {
        file.flush()?;
    }

    println!("File split into {} parts.", file_number - 1);

    remove_file(input_file)?;
    println!("Deleted original file: {}", input_file);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <input_file> <lines_per_file>", args[0]);
        std::process::exit(1);
    }

    let input_file = &args[1];
    let lines_per_file = args[2].parse().map_err(|e| {
        Error::new(
            ErrorKind::InvalidInput,
            format!("Invalid lines_per_file value: {}", e),
        )
    })?;

    match split_large_file(input_file, lines_per_file) {
        Ok(()) => Ok(()),
        Err(e) => {
            if let Some(io_err) = e.downcast_ref::<io::Error>() {
                match io_err.kind() {
                    ErrorKind::NotFound => {
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
