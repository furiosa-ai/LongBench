# This scripts takes two arguments:
# 1. The directory containing JSONL files.
# 2. The output CSV file path.
#
# Usage: python extract_true_answers.py <directory> <output_csv>
#
# This scripts works as follows:
# 1. read a given directory, read all files as pd.DataFrame, and merge them into one DataFrame.
# 2. Find all unique _ids where judge is True.
# 3. Save the unique _ids to a CSV file.
#
# The schema of JSONL files in the directory is:
# ['_id', 'domain', 'sub_domain', 'difficulty', 'length', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'answer', 'context', 'response', 'pred', 'judge']

import sys
import os
import pandas as pd

# Read all JSONL files in the specified directory
def read_jsonl_files(directory):
    dataframes = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        df = pd.read_json(file_path, lines=True)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Extract unique _ids where judge is True
def extract_true_ids(df):
    true_ids = df[df['judge'] == True]['_id'].unique()
    return pd.DataFrame(true_ids, columns=['_id'])

# Save the unique _ids to a CSV file
def save_to_csv(df, output_csv):
    # Drop duplicates if any
    df = df.drop_duplicates()
    df.to_csv(output_csv, header=False, index=False)

# Main function to execute the script
def main(directory, output_csv):
    df = read_jsonl_files(directory)
    true_ids_df = extract_true_ids(df)
    save_to_csv(true_ids_df, output_csv)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_true_answers.py <directory> <output_csv>")
        sys.exit(1)

    directory = sys.argv[1]
    output_csv = sys.argv[2]

    main(directory, output_csv)
    print(f"Unique _ids with judge True saved to {output_csv}")

# End of script
