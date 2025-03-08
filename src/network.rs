use crate::def::{CHESS_SQUARE_COUNT, TERMINATE_SCORE, WHITE};
use crate::network_weights::{
    COMMON_SCALING_FACTOR, HIDDEN_LAYER_BIASES, HIDDEN_LAYER_SIZE,
    HIDDEN_LAYER_TO_OUTPUT_LAYER_WEIGHTS, INPUT_LAYER_SIZE, INPUT_LAYER_TO_HIDDEN_LAYER_WEIGHTS,
    OUTPUT_BIAS,
};
use crate::types::{ChessPiece, ChessSquare, Player, Score};
use crate::util::{FLIPPED_CHESS_SQUARES, MIRRORED_CHESS_PIECES};
use std::f32::consts::E;

pub type NetworkInputValue = i32;
pub type NetworkOutputValue = f32;

const CENTI_PAWN_SCORE_SCALING_FACTOR: f32 = 0.004;

pub const fn calculate_network_input_layer_index(
    chess_piece: ChessPiece,
    chess_square: ChessSquare,
) -> usize {
    (chess_piece as usize - 1) * CHESS_SQUARE_COUNT + chess_square
}

#[inline(always)]
fn relu(x: NetworkInputValue) -> NetworkInputValue {
    x.max(0)
}

#[inline(always)]
fn sigmoid(x: NetworkOutputValue) -> NetworkOutputValue {
    1.0 / (1.0 + E.powf(-x))
}

pub struct Network {
    transposed_input_layer_to_hidden_layer_weights:
        [[NetworkInputValue; HIDDEN_LAYER_SIZE]; INPUT_LAYER_SIZE],
    hidden_layer_biases: [NetworkInputValue; HIDDEN_LAYER_SIZE],
    hidden_layer_to_output_layer_weights: [NetworkInputValue; HIDDEN_LAYER_SIZE],
    output_layer_bias: NetworkOutputValue,

    common_scaling_factor: NetworkOutputValue,

    white_accumulated_layer: [NetworkInputValue; HIDDEN_LAYER_SIZE],
    black_accumulated_layer: [NetworkInputValue; HIDDEN_LAYER_SIZE],
}

impl Network {
    pub fn new() -> Self {
        let mut new_network = Self {
            transposed_input_layer_to_hidden_layer_weights: [[0; HIDDEN_LAYER_SIZE];
                INPUT_LAYER_SIZE],
            hidden_layer_biases: [0; HIDDEN_LAYER_SIZE],
            hidden_layer_to_output_layer_weights: [0; HIDDEN_LAYER_SIZE],
            output_layer_bias: 0.,

            common_scaling_factor: 0.,

            white_accumulated_layer: [0; HIDDEN_LAYER_SIZE],
            black_accumulated_layer: [0; HIDDEN_LAYER_SIZE],
        };

        load_default_weights_and_biases(&mut new_network);
        new_network
    }

    pub fn add(&mut self, chess_piece: ChessPiece, chess_square: ChessSquare) {
        let index_white_perspective =
            calculate_network_input_layer_index(chess_piece, chess_square);

        for (i, accumulated_value) in self.white_accumulated_layer.iter_mut().enumerate() {
            *accumulated_value +=
                self.transposed_input_layer_to_hidden_layer_weights[index_white_perspective][i];
        }

        let index_black_perspective = calculate_network_input_layer_index(
            MIRRORED_CHESS_PIECES[chess_piece as usize],
            FLIPPED_CHESS_SQUARES[chess_square],
        );

        for (i, accumulated_value) in self.black_accumulated_layer.iter_mut().enumerate() {
            *accumulated_value +=
                self.transposed_input_layer_to_hidden_layer_weights[index_black_perspective][i];
        }
    }

    pub fn remove(&mut self, chess_piece: ChessPiece, chess_square: ChessSquare) {
        let index_white_perspective =
            calculate_network_input_layer_index(chess_piece, chess_square);

        for (i, accumulated_value) in self.white_accumulated_layer.iter_mut().enumerate() {
            *accumulated_value -=
                self.transposed_input_layer_to_hidden_layer_weights[index_white_perspective][i];
        }

        let index_black_perspective = calculate_network_input_layer_index(
            MIRRORED_CHESS_PIECES[chess_piece as usize],
            FLIPPED_CHESS_SQUARES[chess_square],
        );

        for (i, accumulated_value) in self.black_accumulated_layer.iter_mut().enumerate() {
            *accumulated_value -=
                self.transposed_input_layer_to_hidden_layer_weights[index_black_perspective][i];
        }
    }

    pub fn clear_accumulated_layer(&mut self) {
        self.white_accumulated_layer = [0; HIDDEN_LAYER_SIZE];
        self.black_accumulated_layer = [0; HIDDEN_LAYER_SIZE];
    }

    pub fn evaluate(&self, player: Player) -> Score {
        let mut hidden_layer = [0; HIDDEN_LAYER_SIZE];

        if player == WHITE {
            for i in 0..HIDDEN_LAYER_SIZE {
                hidden_layer[i] =
                    relu(self.white_accumulated_layer[i] + self.hidden_layer_biases[i]);
            }
        } else {
            for i in 0..HIDDEN_LAYER_SIZE {
                hidden_layer[i] =
                    relu(self.black_accumulated_layer[i] + self.hidden_layer_biases[i]);
            }
        }

        let mut output = self.output_layer_bias;

        for i in 0..HIDDEN_LAYER_SIZE {
            output += (hidden_layer[i] * self.hidden_layer_to_output_layer_weights[i]) as NetworkOutputValue * self.common_scaling_factor;
        }

        win_probability_to_centi_pawn_score(sigmoid(output,
        ))
    }

    fn load_un_flatten(
        &mut self,
        input_layer_to_hidden_layer_weights: Vec<NetworkInputValue>,
        hidden_layer_biases: Vec<NetworkInputValue>,
        hidden_layer_to_output_layer_weights: Vec<NetworkInputValue>,
        output_bias: NetworkOutputValue,
        common_scaling_factor: NetworkOutputValue,
    ) {
        let mut offset = 0;

        let mut transposed_input_layer_to_hidden_layer_weights =
            [[0; INPUT_LAYER_SIZE]; HIDDEN_LAYER_SIZE];

        for i in 0..HIDDEN_LAYER_SIZE {
            for j in 0..INPUT_LAYER_SIZE {
                transposed_input_layer_to_hidden_layer_weights[i][j] =
                    input_layer_to_hidden_layer_weights[offset];
                offset += 1;
            }
        }

        for i in 0..INPUT_LAYER_SIZE {
            for j in 0..HIDDEN_LAYER_SIZE {
                self.transposed_input_layer_to_hidden_layer_weights[i][j] =
                    transposed_input_layer_to_hidden_layer_weights[j][i];
            }
        }

        for i in 0..HIDDEN_LAYER_SIZE {
            self.hidden_layer_biases[i] = hidden_layer_biases[i];
        }

        for i in 0..HIDDEN_LAYER_SIZE {
            self.hidden_layer_to_output_layer_weights[i] = hidden_layer_to_output_layer_weights[i];
        }

        self.output_layer_bias = output_bias;
        self.common_scaling_factor = common_scaling_factor * common_scaling_factor;
    }
}

fn win_probability_to_centi_pawn_score(win_probability: NetworkOutputValue) -> Score {
    (((win_probability.ln() - (1.0 - win_probability).ln()) / CENTI_PAWN_SCORE_SCALING_FACTOR)
        as Score)
        .min(TERMINATE_SCORE - 1)
        .max(-TERMINATE_SCORE + 1)
}

fn load_default_weights_and_biases(network: &mut Network) {
    network.load_un_flatten(
        INPUT_LAYER_TO_HIDDEN_LAYER_WEIGHTS.to_vec(),
        HIDDEN_LAYER_BIASES.to_vec(),
        HIDDEN_LAYER_TO_OUTPUT_LAYER_WEIGHTS.to_vec(),
        OUTPUT_BIAS,
        COMMON_SCALING_FACTOR,
    );
}
