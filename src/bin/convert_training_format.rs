// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, LineWriter, Write};
use std::path::Path;

fn convert_old_to_new_format(
    input_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(output_path).exists() {
        fs::create_dir_all(output_path)?;
    }

    let input_files: Vec<String> = fs::read_dir(input_path)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() {
                path.to_str().map(|s| s.to_string())
            } else {
                None
            }
        })
        .collect();

    if input_files.is_empty() {
        return Err("No files found in input directory".into());
    }

    println!(
        "Converting {} files from old format to new format...",
        input_files.len()
    );

    let mut total_lines_converted = 0;
    let mut total_errors = 0;

    for input_file in &input_files {
        let filename = Path::new(input_file)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");
        let output_file = format!("{}/{}", output_path, filename);

        println!("Processing {} -> {}...", input_file, output_file);

        let (lines_converted, errors) = convert_file(input_file, &output_file)?;
        total_lines_converted += lines_converted;
        total_errors += errors;
    }

    println!("\nConversion complete!");
    println!("Total lines converted: {}", total_lines_converted);
    println!("Total errors: {}", total_errors);
    println!("Output saved to: {}", output_path);

    Ok(())
}

fn convert_file(
    input_file: &str,
    output_file: &str,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);

    let output_file = File::create(output_file)?;
    let mut writer = LineWriter::new(output_file);

    let mut lines_converted = 0;
    let mut errors = 0;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        match convert_line(&line) {
            Ok(converted_line) => {
                writer.write_all(converted_line.as_bytes())?;
                writer.write_all(b"\n")?;
                lines_converted += 1;
            }
            Err(e) => {
                eprintln!("  Error on line {}: {}", line_num + 1, e);
                errors += 1;
            }
        }
    }

    writer.flush()?;
    Ok((lines_converted, errors))
}

fn convert_line(line: &str) -> Result<String, Box<dyn std::error::Error>> {
    let parts: Vec<&str> = line.split(',').collect();

    if parts.len() < 3 {
        return Err(format!("Line has too few parts: {}", line).into());
    }

    let result_str = parts[parts.len() - 2];

    let result = match result_str {
        "0" => "0.0",
        "1" => "1.0",
        "0.5" => "0.5",
        _ => {
            let result_float: f64 = result_str
                .parse()
                .map_err(|_| format!("Invalid result '{}'", result_str))?;

            if result_float == 0.0 {
                "0.0"
            } else if result_float == 1.0 {
                "1.0"
            } else if result_float == 0.5 {
                "0.5"
            } else {
                return Err(format!("Result must be 0, 0.5, or 1, got {}", result_float).into());
            }
        }
    };

    let feature_parts = &parts[..parts.len() - 2];

    let mut new_features = String::new();
    let mut zero_count = 0;

    for part in feature_parts {
        if *part == "X" {
            if zero_count > 0 {
                new_features.push_str(&zero_count.to_string());
                zero_count = 0;
            }
            new_features.push('X');
        } else {
            let count: u32 = part
                .parse()
                .map_err(|_| format!("Invalid zero count '{}'", part))?;
            zero_count += count;
        }
    }

    if zero_count > 0 {
        new_features.push_str(&zero_count.to_string());
    }

    Ok(format!("{},{}", new_features, result))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <input_dir> <output_dir>", args[0]);
        eprintln!("Example: {} old_training_data new_training_data", args[0]);
        std::process::exit(1);
    }

    let input_dir = &args[1];
    let output_dir = &args[2];

    if !Path::new(input_dir).exists() {
        eprintln!("Error: Input directory '{}' does not exist.", input_dir);
        std::process::exit(1);
    }

    match convert_old_to_new_format(input_dir, output_dir) {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
