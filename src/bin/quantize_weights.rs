// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

const INPUT_LAYER_SIZE: usize = 768;

fn quantize_to_int8_with_scale(weights: &[f32]) -> (Vec<i8>, f32) {
    if weights.is_empty() {
        return (Vec::new(), 1.0);
    }

    let max_abs = weights
        .iter()
        .copied()
        .fold(0.0f32, |acc, w| acc.max(w.abs()));

    let qmax = i8::MAX as f32;

    let scale = if max_abs > 1e-6 { max_abs / qmax } else { 1.0 };

    let quantized: Vec<i8> = weights
        .iter()
        .map(|&w| {
            let q = w / scale;
            q.round().clamp(i8::MIN as f32, i8::MAX as f32) as i8
        })
        .collect();

    (quantized, scale)
}

fn parse_float_array_from_csv(csv_path: &Path) -> io::Result<Vec<f32>> {
    let file = File::open(csv_path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let first_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "CSV file is empty"))??;

    let values: Vec<f32> = first_line
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.trim().parse::<f32>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed to parse float: {}", e),
                )
            })
        })
        .collect::<Result<_, _>>()?;

    Ok(values)
}

fn export_quantized_weights(
    fc1_weights: &[f32],
    fc1_bias: &[f32],
    fc2_weights: &[f32],
    fc2_bias: f32,
    export_file: &Path,
) -> io::Result<()> {
    let (fc1_quantized, scale) = quantize_to_int8_with_scale(fc1_weights);

    let mut writer = BufWriter::new(File::create(export_file)?);

    for val in &fc1_quantized {
        write!(writer, "{}", val)?;
        write!(writer, ",")?;
    }

    for val in fc1_bias {
        write!(writer, "{}", val)?;
        write!(writer, ",")?;
    }

    for val in fc2_weights {
        write!(writer, "{}", val)?;
        write!(writer, ",")?;
    }

    write!(writer, "{}", fc2_bias)?;
    write!(writer, ",")?;

    write!(writer, "{}", scale)?;

    writeln!(writer)?;

    writer.flush()?;

    println!(
        "Quantized weights exported to {} (scale: {})",
        export_file.display(),
        scale
    );

    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!(
            "Usage: {} <hidden_layer_size> <input_file> <export_file>",
            args[0]
        );
        std::process::exit(1);
    }

    let hidden_layer_size: usize = args[1]
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid hidden_layer_size"))?;

    let input_file = Path::new(&args[2]);
    if !input_file.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Input file '{}' does not exist", input_file.display()),
        ));
    }

    let export_file = Path::new(&args[3]);

    println!("Loading weights from {}...", input_file.display());

    let data = parse_float_array_from_csv(input_file)?;

    let fc1_weight_count = INPUT_LAYER_SIZE * hidden_layer_size;
    let fc1_bias_count = hidden_layer_size;
    let fc2_weight_count = hidden_layer_size;
    let fc2_bias_count = 1;

    let expected_total = fc1_weight_count + fc1_bias_count + fc2_weight_count + fc2_bias_count;

    if data.len() != expected_total {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Input file has wrong number of elements: expected {}, got {}",
                expected_total,
                data.len()
            ),
        ));
    }

    let mut offset = 0;

    let fc1_weights = &data[offset..offset + fc1_weight_count];
    offset += fc1_weight_count;

    let fc1_bias = &data[offset..offset + fc1_bias_count];
    offset += fc1_bias_count;

    let fc2_weights = &data[offset..offset + fc2_weight_count];
    offset += fc2_weight_count;

    let fc2_bias = data[offset];

    export_quantized_weights(fc1_weights, fc1_bias, fc2_weights, fc2_bias, export_file)?;

    Ok(())
}
