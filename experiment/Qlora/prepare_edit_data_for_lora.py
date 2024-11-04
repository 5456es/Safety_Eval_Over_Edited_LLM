import json
import os
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_part", type=int, default=0)
    parser.add_argument("--data_source", type=str, default='ZsRE')
    parser.add_argument("--data_size", type=int, default=1)
    parser.add_argument("--random", type=bool, default=False)

    args = parser.parse_args()

    # Debug: Print input arguments
    print(f"Arguments received: data_part={args.data_part}, data_source={args.data_source}, "
          f"data_size={args.data_size}, random={args.random}")

    # Assertions to check the validity of input arguments
    assert args.data_part in [0, 1, 2], "data_part must be 0, 1, or 2."
    assert args.data_source in ['ZsRE', 'wiki_recent', 'wiki_counterfact', 'NEWS2024'], \
        "data_source must be one of 'ZsRE', 'wiki_recent', 'wiki_counterfact', 'NEWS2024'."
    assert args.data_size >= 1, "data_size must be at least 1."

    # Load data from file
    file_path = f'../../data/edit_data/merged_data_part_{args.data_part}.json'
    print(f"Loading data from {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries from the data file.")
    except FileNotFoundError:
        print(f"Error: File not found at path {file_path}")
        exit(1)

    # Filter data based on the source
    data = [entry for entry in data if entry['source'] == args.data_source]
    print(f"Filtered data based on source '{args.data_source}', resulting in {len(data)} entries.")

    # Random shuffle the data if specified
    if args.random:
        print("Shuffling data randomly.")
        random.shuffle(data)

    # Limit data to the specified data size
    data = data[:args.data_size]
    print(f"Selected {len(data)} entries based on data size.")

    # Convert data to the specified format and save to a .jsonl file
    output_path = 'tmp_data.jsonl'
    print(f"Saving formatted data to {output_path}")
    with open(output_path, 'w') as f:
        for entry in data:
            json_line = json.dumps({"instruction": entry['prompt'], "output": entry['target_new']})
            f.write(json_line + '\n')

    print(f"Data successfully saved to {output_path}")
