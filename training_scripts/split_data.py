import os
import argparse

DEFAULT_LINES_PER_FILE=2000000

def split_large_file(input_file, lines_per_file):
    try:
        with open(input_file, 'r') as file:
            file_number = 1
            line_count = 0
            output_file = None

            for line in file:
                if line_count % lines_per_file == 0:
                    if output_file:
                        output_file.close()
                    output_file = open(f'{input_file}_part_{file_number}.txt', 'w')
                    file_number += 1

                output_file.write(line)
                line_count += 1

            if output_file:
                output_file.close()

        print(f"File split into {file_number - 1} parts.")

        os.remove(input_file)
        print(f"Deleted original file: {input_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a large file into smaller parts.")
    parser.add_argument("input_file", help="Path to the input file to split.")
    parser.add_argument("--lines_per_file", type=int, default=DEFAULT_LINES_PER_FILE,
                        help="Number of lines per output file.")
    args = parser.parse_args()

    split_large_file(args.input_file, args.lines_per_file)
