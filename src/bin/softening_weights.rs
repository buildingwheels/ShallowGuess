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
use std::fs::{self, File};
use std::io::{LineWriter, Write};
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

use shallow_guess::util::{
    read_lines,
};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 5 {
        eprintln!(
            "Usage: {} <input_dir> <output_dir> <batch_size> <num_threads>",
            args[0]
        );
        std::process::exit(1);
    }

    let input_dir = &args[1];
    let output_dir = PathBuf::from(&args[2]);
    let batch_size = args[3].parse::<usize>().unwrap();

    fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    let files: Vec<_> = fs::read_dir(input_dir)
        .expect("Failed to read input directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .collect();

    if files.is_empty() {
        println!("No files found in input directory");
        return;
    }

    let num_threads = args[4].parse::<usize>().unwrap().max(1).min(files.len());

    let (tx, rx) = mpsc::channel::<(PathBuf, PathBuf)>();
    let rx = Arc::new(Mutex::new(rx));

    thread::scope(|s| {
        for _ in 0..num_threads {
            let rx = Arc::clone(&rx);
            s.spawn(move || loop {
                let task = {
                    let rx = rx.lock().unwrap();
                    rx.recv()
                };
                match task {
                    Ok((input_path, output_path)) => {
                        softening_weights(
                            input_path.to_str().unwrap(),
                            output_path.to_str().unwrap(),
                            batch_size,
                        );
                    }
                    Err(_) => break,
                }
            });
        }

        for entry in files {
            let input_path = entry.path();
            let output_path = output_dir.join(entry.file_name());
            tx.send((input_path, output_path)).unwrap();
        }

        drop(tx);
    });
}

fn softening_weights(training_data_file: &str, output_file_path: &str, batch_count: usize) {
    let output_file = File::create(output_file_path).unwrap();
    let mut file_writer = LineWriter::new(output_file);

    let mut training_size = 0;
    let mut buffered_line_count = 0;
    let mut output_batch_buffer = String::new();

    let start_time = Instant::now();

    if let Ok(lines) = read_lines(training_data_file) {
        for line in lines.flatten() {
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            let splits = line.split(',').collect::<Vec<&str>>();
            let features = splits[0].trim();
            let mut result: f32 = splits[1].parse().unwrap();

            let distance_to_termination_ratio = (result - 0.5).abs() / 0.5;

            if result > 0.5 {
                result = 0.5 + 0.5 * distance_to_termination_ratio * distance_to_termination_ratio;
            } else if result < 0.5 {
                result = 0.5 - 0.5 * distance_to_termination_ratio * distance_to_termination_ratio;
            }

            output_batch_buffer.push_str(&format!("{},{}\n", features, result));
            training_size += 1;
            buffered_line_count += 1;

            if buffered_line_count > batch_count {
                println!(
                    "Processed {} positions with PPS {}",
                    training_size,
                    training_size / start_time.elapsed().as_secs().max(1)
                );

                file_writer.write(output_batch_buffer.as_bytes()).unwrap();
                output_batch_buffer.clear();
                buffered_line_count = 0;
            }
        }

        file_writer.write(output_batch_buffer.as_bytes()).unwrap();
    }

    println!(
        "Completed updating training set with size {} to: {}, time taken {}seconds",
        training_size,
        output_file_path,
        start_time.elapsed().as_secs()
    );

    println!("Final updated training set size: {}", training_size);
}
