import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import concurrent.futures
import time
from tqdm import tqdm


def process_file(filepath, output_dir):
    """
    Process a GFZ loading file and save it as a pickle using pandas.

    Args:
        filepath (str): Path to the input file
        output_dir (str): Directory to save the pickle file
    """
    try:
        # Extract filename
        filename = os.path.basename(filepath)

        # Extract 4-char code (first 4 characters)
        charcode = filename[:4]

        # Determine component based on the file naming pattern
        if 'ntal' in filename:
            component = 'A'
        elif 'ntol' in filename:
            component = 'O'
        elif 'slel' in filename:
            component = 'S'
        elif 'cwsl' in filename:
            component = 'H'
        else:
            print(f"Unknown component in file: {filename}")
            return None

        # Determine frame (cf or cm)
        if '.cf' in filename:
            frame = 'cf'
        elif '.cm' in filename:
            frame = 'cm'
        else:
            print(f"Unknown frame in file: {filename}")
            return None

        # Read header lines
        header = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    header.append(line.strip())
                else:
                    break

        # Read data with pandas
        df = pd.read_csv(
            filepath,
            comment='#',
            delim_whitespace=True,
            header=None,
            names=['year', 'month', 'day', 'hour', 'R', 'EW', 'NS']
        )

        # Create datetime column
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

        # Reorder columns to match the requested format
        df = df[['datetime', 'NS', 'EW', 'R']].set_index('datetime')

        # Create output filename
        output_filename = f"{charcode}_{component}_{frame}.pkl"
        output_path = os.path.join(output_dir, output_filename)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(df, f)

        return f"Processed {filename} -> {output_filename}"
    except Exception as e:
        return f"Error processing {os.path.basename(filepath)}: {str(e)}"


def main():
    """
    Process all GFZ loading files in the specified directory using multi-threading.
    """
    start_time = time.time()

    # Directory containing input files
    # input_dir = os.path.dirname(os.path.abspath(__file__))
    # Uncomment and modify this line if you want to use a specific input directory
    input_dir = r'D:\from_Kyriakos\non_tidal_loading_station-wise_igs_repro3_extension'

    # Output directory for pickle files
    output_dir = "EXT\\ESMGFZLOADING\\CODE2"

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all relevant files
    files_to_process = []
    for filename in os.listdir(input_dir):
        # Look for files with the pattern like 00NA-59975M001.lsdm.gfz13.cwsl.cf
        if re.match(r'^\w{4}-.*\.(ntal|ntol|slel|cwsl)\.(cf|cm)$', filename):
            filepath = os.path.join(input_dir, filename)
            files_to_process.append(filepath)

    print(f"Found {len(files_to_process)} files to process")

    # Determine the number of workers (threads)
    # Use max(1, os.cpu_count() - 1) to leave one CPU for system tasks
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Using {num_workers} worker threads")

    # Process files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, filepath, output_dir): filepath
                          for filepath in files_to_process}

        # Create a progress bar
        with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        # Log result but don't print to console to avoid cluttering with many threads
                        # The progress bar will show progress instead
                        pass
                except Exception as exc:
                    print(f"{os.path.basename(filepath)} generated an exception: {exc}")

                # Update progress bar
                pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"Processed {len(files_to_process)} files in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()