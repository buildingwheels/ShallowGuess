import os
import sys
import random
import argparse
import contextlib    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blend and split input files into multiple output files.")
    parser.add_argument("input_path", help="Directory containing input files.")
    parser.add_argument("output_path", help="Directory to save output files.")
    parser.add_argument("output_num_files", type=int, help="Number of output files to create.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_path):
        print(f"Error: Input directory '{args.input_path}' does not exist.")
        sys.exit(1)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    input_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) 
                  if os.path.isfile(os.path.join(args.input_path, f))]

    if not input_files:
        print(f"Error: No files found in input directory '{args.input_path}'.")
        sys.exit(1)

    output_files = [os.path.join(args.output_path, f'mixed_{i+1}.tempdata') 
                    for i in range(args.output_num_files)]

    try:
        with contextlib.ExitStack() as stack:
            output_handles = [stack.enter_context(open(file, 'w')) for file in output_files]

            for input_file in input_files:
                with open(input_file, 'r') as infile:
                    for line in infile:
                        random_output_handle = random.choice(output_handles)
                        random_output_handle.write(line)

        print(f"Written blended data into {output_files}")

        for i, output_file in enumerate(output_files):
            new_name = os.path.join(args.output_path, f'{i+1}.nninput')
            os.rename(output_file, new_name)

        print("Renamed all files.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
