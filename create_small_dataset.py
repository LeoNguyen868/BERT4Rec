import os

def create_small_dataset(original_file="data/ml-1m.txt",
                         small_file_path="data/ml-1m-small.txt",
                         num_lines=1000):
    """
    Creates a smaller dataset by taking the first num_lines from the original.
    """
    if not os.path.exists("data"):
        os.makedirs("data")
        print(f"Created directory: data")

    print(f"Reading first {num_lines} lines from {original_file}...")
    lines_written = 0
    try:
        with open(original_file, "r") as f_orig, open(small_file_path, "w") as f_small:
            for i, line in enumerate(f_orig):
                if i < num_lines:
                    f_small.write(line)
                    lines_written += 1
                else:
                    break
        print(f"Successfully wrote {lines_written} lines to {small_file_path}")
        if lines_written < num_lines:
            print(f"Warning: Original file had less than {num_lines} lines.")
    except FileNotFoundError:
        print(f"Error: Original data file {original_file} not found.")
        print("Please ensure the dataset is in the correct location.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_small_dataset() 