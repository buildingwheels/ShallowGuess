import os
import sys
import random

input_path = sys.argv[1]
output_num_files = int(sys.argv[2])

output_path = input_path

input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

output_files = [os.path.join(output_path, f'mixed_{i+1}.txt') for i in range(output_num_files)]

output_handles = [open(file, 'w') for file in output_files]

for input_file in input_files:
    with open(input_file, 'r') as infile:
        for line in infile:
            random_output_handle = random.choice(output_handles)
            random_output_handle.write(line)

for handle in output_handles:
    handle.close()

print(f"Written blended data into {output_files}")

for input_file in input_files:
    os.remove(input_file)
    print(f"Deleted original file: {input_file}")

for i, output_file in enumerate(output_files):
    new_name = os.path.join(output_path, f'{i+1}.txt')
    os.rename(output_file, new_name)

print("Renamed all files")
