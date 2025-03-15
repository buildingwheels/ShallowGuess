import os
import sys

def compress_line(line):
    parts = line.strip().split(',')
    compressed = []
    count = 0

    for part in parts[:-1]:
        if part == '0':
            count += 1
        else:
            if count > 0:
                compressed.append(str(count))
                count = 0
            compressed.append('X')

    if count > 0:
        compressed.append(str(count))
    compressed.append(parts[-1])

    return ','.join(compressed)

def compress_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            compressed_line = compress_line(line)
            outfile.write(compressed_line + '\n')

def compress_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        compress_file(input_file, output_file)
        print(f"Compressed {input_file} to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compress.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    compress_directory(input_directory, output_directory)
