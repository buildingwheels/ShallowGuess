import os
import sys

def split_large_file(input_file, lines_per_file):
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


input_file = sys.argv[1]

split_large_file(input_file, lines_per_file=2000000)
