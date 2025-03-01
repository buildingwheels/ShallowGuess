use std::fs;
use std::io::Write;

fn main() {
    load_weights(768, 128);
}

fn load_weights(input_layer_size: usize, hidden_layer_size: usize) {
    let weights_file = &format!("resources/weights/1L-{}.weights", hidden_layer_size);
    let out_file = "src/network_weights.rs";

    let file_content =
        fs::read_to_string(weights_file).expect("Failed to read the model weights file");

    let values: Vec<f32> = file_content
        .split(',')
        .map(|s| s.trim().parse::<f32>().unwrap())
        .collect();

    let input_layer_to_hidden_layer_size = input_layer_size * hidden_layer_size;

    let input_layer_to_hidden_layer_weights = &values[0..input_layer_to_hidden_layer_size];
    let hidden_layer_biases = &values
        [input_layer_to_hidden_layer_size..input_layer_to_hidden_layer_size + hidden_layer_size];

    let hidden_layer_to_output_layer_weights = &values[input_layer_to_hidden_layer_size
        + hidden_layer_size
        ..input_layer_to_hidden_layer_size + hidden_layer_size + hidden_layer_size];
    let output_bias =
        values[input_layer_to_hidden_layer_size + hidden_layer_size + hidden_layer_size];

    let hidden_layer_scaling_factor =
        values[input_layer_to_hidden_layer_size + hidden_layer_size + hidden_layer_size + 1];
    let output_layer_scaling_factor =
        values[input_layer_to_hidden_layer_size + hidden_layer_size + hidden_layer_size + 2];

    let mut code = String::new();
    code.push_str(&format!(
        "pub const INPUT_LAYER_SIZE: usize = {};\n",
        input_layer_size
    ));
    code.push_str(&format!(
        "pub const HIDDEN_LAYER_SIZE: usize = {};\n",
        hidden_layer_size
    ));

    code.push_str(&format!(
        "pub const INPUT_LAYER_TO_HIDDEN_LAYER_WEIGHTS: [i32; {}] = [\n",
        input_layer_to_hidden_layer_size
    ));
    for &value in input_layer_to_hidden_layer_weights {
        code.push_str(&format!("{}, ", value as i32));
    }
    code.push_str("];\n\n");

    code.push_str(&format!(
        "pub const HIDDEN_LAYER_BIASES: [f32; {}] = [\n",
        hidden_layer_size
    ));
    for &value in hidden_layer_biases {
        code.push_str(&format!("{}, ", value));
    }
    code.push_str("];\n\n");

    code.push_str(&format!(
        "pub const HIDDEN_LAYER_TO_OUTPUT_LAYER_WEIGHTS: [i32; {}] = [\n",
        hidden_layer_size
    ));
    for &value in hidden_layer_to_output_layer_weights {
        code.push_str(&format!("{}, ", value as i32));
    }
    code.push_str("];\n\n");

    code.push_str(&format!("pub const OUTPUT_BIAS: f32 = {};\n", output_bias));
    code.push_str(&format!(
        "pub const HIDDEN_LAYER_SCALING_FACTOR: f32 = {};\n",
        hidden_layer_scaling_factor
    ));
    code.push_str(&format!(
        "pub const OUTPUT_LAYER_SCALING_FACTOR: f32 = {};\n",
        output_layer_scaling_factor
    ));

    let mut file = fs::File::create(out_file).expect("Failed to create weights source file");
    file.write_all(code.as_bytes())
        .expect("Failed to write weights source file");
}
