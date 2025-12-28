// Copyright (c) 2025 Zixiao Han
// SPDX-License-Identifier: MIT

use std::collections::HashMap;
use std::fs;
use std::io::Write;

const FIXED_INPUT_LAYER_SIZE: usize = 768;
const FIXED_OUTPUT_LAYER_SIZE: usize = 3;

fn main() {
    let config_content =
        fs::read_to_string("config/network.cfg").expect("Failed to read config file");

    let config: HashMap<String, usize> = config_content
        .lines()
        .filter_map(|line| {
            let mut parts = line.split('=');
            let key = parts.next()?.trim().to_string();
            let value = parts.next()?.trim().parse().ok()?;
            Some((key, value))
        })
        .collect();

    let hidden_layer_size = config
        .get("hidden_layer_size")
        .expect("hidden_layer_size not found in config");

    load_weights(*hidden_layer_size);
}

fn load_weights(hidden_layer_size: usize) {
    let weights_file = &format!("resources/weights/{}.weights", hidden_layer_size);
    let out_file = "src/network_weights.rs";

    let file_content =
        fs::read_to_string(weights_file).expect("Failed to read the model weights file");

    let values: Vec<f32> = file_content
        .split(',')
        .map(|s| s.trim().parse::<f32>().unwrap())
        .collect();

    let input_to_hidden_size = FIXED_INPUT_LAYER_SIZE * hidden_layer_size;
    let hidden_biases_size = hidden_layer_size;
    let hidden_to_output_size = hidden_layer_size * FIXED_OUTPUT_LAYER_SIZE;
    let output_biases_size = FIXED_OUTPUT_LAYER_SIZE;

    let expected_total_size =
        input_to_hidden_size + hidden_biases_size + hidden_to_output_size + output_biases_size + 1;
    assert_eq!(
        values.len(),
        expected_total_size,
        "Unmatched weights size, expected {}, got {}",
        expected_total_size,
        values.len()
    );

    let mut start = 0;
    let input_layer_to_hidden_layer_weights = &values[start..start + input_to_hidden_size];
    start += input_to_hidden_size;

    let hidden_layer_biases = &values[start..start + hidden_biases_size];
    start += hidden_biases_size;

    let hidden_layer_to_output_layer_weights = &values[start..start + hidden_to_output_size];
    start += hidden_to_output_size;

    let output_biases = &values[start..start + output_biases_size];
    start += output_biases_size;

    let scaling_factor = values[start];

    let mut code = String::new();
    code.push_str(&format!(
        "pub const INPUT_LAYER_SIZE: usize = {};\n",
        FIXED_INPUT_LAYER_SIZE
    ));
    code.push_str(&format!(
        "pub const HIDDEN_LAYER_SIZE: usize = {};\n",
        hidden_layer_size
    ));
    code.push_str(&format!(
        "pub const OUTPUT_LAYER_SIZE: usize = {};\n",
        FIXED_OUTPUT_LAYER_SIZE
    ));

    code.push_str(&format!(
        "pub const INPUT_LAYER_TO_HIDDEN_LAYER_WEIGHTS: [i16; {}] = [\n",
        input_to_hidden_size
    ));
    for &value in input_layer_to_hidden_layer_weights {
        code.push_str(&format!("{}, ", value));
    }
    code.push_str("];\n\n");

    code.push_str(&format!(
        "pub const HIDDEN_LAYER_BIASES: [f32; {}] = [\n",
        hidden_biases_size
    ));
    for &value in hidden_layer_biases {
        code.push_str(&format!("{}, ", string_float(value)));
    }
    code.push_str("];\n\n");

    code.push_str(&format!(
        "pub const HIDDEN_LAYER_TO_OUTPUT_LAYER_WEIGHTS: [[f32; {}]; {}] = [\n",
        hidden_layer_size, FIXED_OUTPUT_LAYER_SIZE
    ));

    for output_idx in 0..FIXED_OUTPUT_LAYER_SIZE {
        code.push_str("    [");
        for hidden_idx in 0..hidden_layer_size {
            let weight_idx = output_idx * hidden_layer_size + hidden_idx;
            let value = hidden_layer_to_output_layer_weights[weight_idx];
            code.push_str(&format!("{}, ", string_float(value)));
        }
        code.push_str("],\n");
    }
    code.push_str("];\n\n");

    code.push_str(&format!(
        "pub const OUTPUT_BIASES: [f32; {}] = [\n",
        output_biases_size
    ));
    for &value in output_biases {
        code.push_str(&format!("{}, ", string_float(value)));
    }
    code.push_str("];\n\n");

    code.push_str(&format!(
        "pub const SCALING_FACTOR: f32 = {};\n",
        scaling_factor
    ));

    let mut file = fs::File::create(out_file).expect("Failed to create weights source file");
    file.write_all(code.as_bytes())
        .expect("Failed to write weights source file");
}

fn string_float(value: f32) -> String {
    if value == 0. {
        format!("{}.0", value)
    } else {
        format!("{}", value)
    }
}
