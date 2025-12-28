// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

use shallow_guess::prng::RandGenerator;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn blend_data(
    input_path: &str,
    output_path: &str,
    num_files: usize,
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

    let mut output_files = Vec::new();
    let mut output_handles = Vec::new();

    for i in 0..num_files {
        let filename = format!("{}/mixed_{}.tempdata", output_path, i + 1);
        let file = File::create(&filename)?;
        output_files.push(filename);
        output_handles.push(file);
    }

    println!("Processing {} input files...", input_files.len());

    let mut output_line_counts = vec![0; num_files];
    let mut total_lines = 0;

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let mut rng = RandGenerator::new(seed);

    for input_file in &input_files {
        println!("Processing {}...", input_file);
        let file = File::open(input_file)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;

            let output_index = (rng.next_f64() * num_files as f64).floor() as usize;
            output_handles[output_index].write_all(line.as_bytes())?;
            output_handles[output_index].write_all(b"\n")?;
            output_line_counts[output_index] += 1;
            total_lines += 1;
        }
    }

    for handle in &mut output_handles {
        handle.flush()?;
    }

    println!(
        "Processed {} lines from {} files.",
        total_lines,
        input_files.len()
    );
    println!("Written data into {:?}", output_files);

    println!("\nDistribution statistics:");
    for (i, count) in output_line_counts.iter().enumerate() {
        println!("  Output file {}: {} lines", i + 1, count);
    }

    for (i, temp_file) in output_files.iter().enumerate() {
        let new_name = format!("{}/{}", output_path, i + 1);
        fs::rename(temp_file, &new_name)?;
    }

    println!("Renamed all files.");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!("Usage: {} <input_path> <output_path> <num_files>", args[0]);
        eprintln!("Example: {} input_data output_data 5", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let num_files: usize = args[3]
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid number of files"))?;

    if !Path::new(input_path).exists() {
        eprintln!("Error: Input directory '{}' does not exist.", input_path);
        std::process::exit(1);
    }

    match blend_data(input_path, output_path, num_files) {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
