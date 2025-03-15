use std::fs::File;
use std::io::{BufRead, LineWriter, Write};
use std::path::Path;
use std::time::Instant;
use std::{env, io};

use shallow_guess::chess_move_gen::{
    generate_captures_and_promotions, is_in_check, is_invalid_position,
};
use shallow_guess::chess_position::ChessPosition;
use shallow_guess::def::{A1, BLACK, H8, NO_PIECE, PIECE_TYPE_COUNT, WHITE};
use shallow_guess::network::{calculate_network_input_layer_index, Network};
use shallow_guess::network_weights::INPUT_LAYER_SIZE;
use shallow_guess::types::{ChessPiece, ChessSquare, Score, SearchPly};
use shallow_guess::util::{NetworkInputs, FLIPPED_CHESS_SQUARES, MIRRORED_CHESS_PIECES};

const ONE_SYMBOL: &str = "X,";

const MAX_PLY: SearchPly = 8;

const MATERIAL_SCORES: [Score; PIECE_TYPE_COUNT] =
    [0, 1, 3, 3, 5, 10, 100, -1, -3, -3, -5, -10, -100];

fn main() {
    let mut args = env::args().into_iter();
    args.next().unwrap();

    let original_fen_file = args.next().unwrap();
    let processed_fen_file = args.next().unwrap();
    let output_file = args.next().unwrap();
    let skip_count = args.next().unwrap().parse::<usize>().unwrap();
    let max_count = args.next().unwrap().parse::<usize>().unwrap();
    let batch_size = args.next().unwrap().parse::<usize>().unwrap();

    generate_raw_training_set(
        &original_fen_file,
        &processed_fen_file,
        skip_count,
        max_count,
    );
    generate_training_set(&processed_fen_file, &output_file, batch_size);
}

fn generate_training_set(raw_training_set_file: &str, output_file_path: &str, batch_count: usize) {
    let output_file = File::create(output_file_path).unwrap();
    let mut file_writer = LineWriter::new(output_file);

    let mut training_size = 0;
    let mut buffered_line_count = 0;
    let mut output_batch_buffer = String::new();

    let start_time = Instant::now();

    if let Ok(lines) = read_lines(raw_training_set_file) {
        for line in lines.flatten() {
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            let splits = line.split(',').collect::<Vec<&str>>();
            let fen = splits[0];

            let mut chess_position = ChessPosition::new(Network::new());
            chess_position.set_from_fen(fen);

            if is_in_check(&chess_position, chess_position.player) {
                continue;
            }

            let static_score = get_material_score(&chess_position);
            let q_score = exchange_search(&mut chess_position, static_score, static_score + 1, 0);

            if q_score != static_score {
                continue;
            }

            let network_inputs = parse_network_inputs_from_fen(&chess_position);
            let mut result = splits[1].parse::<f64>().unwrap();

            if chess_position.player == BLACK {
                result = 1. - result;
            }

            let mut zero_count = 0;

            for x in network_inputs {
                if x == 0 {
                    zero_count += 1;
                } else {
                    if zero_count > 0 {
                        output_batch_buffer.push_str(&format!("{},", zero_count));
                        zero_count = 0;
                    }

                    output_batch_buffer.push_str(ONE_SYMBOL);
                }
            }

            if zero_count > 0 {
                output_batch_buffer.push_str(&format!("{},", zero_count));
            }

            output_batch_buffer.push_str(&format!("{}\n", result));
            training_size += 1;
            buffered_line_count += 1;

            if buffered_line_count > batch_count {
                file_writer.write(output_batch_buffer.as_bytes()).unwrap();
                output_batch_buffer.clear();
                buffered_line_count = 0;
            }
        }

        file_writer.write(output_batch_buffer.as_bytes()).unwrap();
    }

    println!(
        "Completed generating training set with size {} to: {}, time taken {}seconds",
        training_size,
        output_file_path,
        start_time.elapsed().as_secs()
    );
}

fn generate_raw_training_set(
    fen_file: &str,
    output_file_path: &str,
    skip_opening_positions: usize,
    max_positions_per_game: usize,
) {
    let output_file = File::create(output_file_path).unwrap();
    let mut file_writer = LineWriter::new(output_file);
    let mut current_game_result = 0.;
    let mut position_count = 0;

    if let Ok(lines) = read_lines(fen_file) {
        for line in lines.flatten() {
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            if line.contains("Result") {
                if line.contains("1/2") {
                    current_game_result = 0.5;
                } else if line.contains("0-1") {
                    current_game_result = 0.;
                } else if line.contains("1-0") {
                    current_game_result = 1.;
                }

                position_count = 0;
                continue;
            } else if !line.contains("[") {
                position_count += 1;

                if position_count > skip_opening_positions
                    && position_count < max_positions_per_game
                {
                    file_writer
                        .write(format!("{},{}\n", line.trim(), current_game_result).as_bytes())
                        .unwrap();
                }
            }
        }
    }

    println!(
        "Completed generating raw training set to: {}",
        output_file_path
    );
}

fn exchange_search(
    chess_position: &mut ChessPosition,
    mut alpha: Score,
    beta: Score,
    ply: SearchPly,
) -> Score {
    let static_eval = get_material_score(chess_position);

    if ply > MAX_PLY {
        return static_eval;
    }

    if static_eval >= beta {
        return beta;
    }

    if static_eval > alpha {
        alpha = static_eval;
    }

    let mut captures_and_promotions = generate_captures_and_promotions(chess_position);

    while let Some(chess_move) = captures_and_promotions.pop() {
        let saved_state = chess_position.make_move(&chess_move);

        if is_invalid_position(chess_position) {
            chess_position.unmake_move(&chess_move, saved_state);
            continue;
        }

        let score = -exchange_search(chess_position, -beta, -alpha, ply + 1);

        chess_position.unmake_move(&chess_move, saved_state);

        if score >= beta {
            return beta;
        }

        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

fn parse_network_inputs_from_fen(chess_position: &ChessPosition) -> NetworkInputs {
    let mut network_inputs = vec![0; INPUT_LAYER_SIZE];

    if chess_position.player == WHITE {
        for current_square in A1..=H8 {
            let piece = chess_position.board[current_square];

            if piece == NO_PIECE {
                continue;
            }

            update_training_network_inputs(&mut network_inputs, piece, current_square);
        }
    } else {
        for current_square in A1..=H8 {
            let piece = chess_position.board[current_square];

            if piece == NO_PIECE {
                continue;
            }

            update_mirrored_training_network_inputs(&mut network_inputs, piece, current_square);
        }
    }

    network_inputs
}

fn get_material_score(chess_position: &ChessPosition) -> Score {
    let mut score = 0;

    for chess_square in A1..=H8 {
        score += MATERIAL_SCORES[chess_position.board[chess_square] as usize];
    }

    if chess_position.player == WHITE {
        score
    } else {
        -score
    }
}

fn update_training_network_inputs(
    network_inputs: &mut NetworkInputs,
    chess_piece: ChessPiece,
    chess_square: ChessSquare,
) {
    network_inputs[calculate_network_input_layer_index(chess_piece, chess_square)] = 1;
}

fn update_mirrored_training_network_inputs(
    network_inputs: &mut NetworkInputs,
    chess_piece: ChessPiece,
    chess_square: ChessSquare,
) {
    network_inputs[calculate_network_input_layer_index(
        MIRRORED_CHESS_PIECES[chess_piece as usize],
        FLIPPED_CHESS_SQUARES[chess_square],
    )] = 1;
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
