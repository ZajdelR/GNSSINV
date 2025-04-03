import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import concurrent.futures
import time
from tqdm import tqdm
import logging
import sys


def setup_logging(log_dir):
    """
    Set up logging to both console and file

    Args:
        log_dir (str): Directory for log files

    Returns:
        logger: Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Get script name without extension
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create log filename with format LOG_@scriptname_@time
    log_filename = f"LOG_{script_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_filepath}")

    return logger


def process_file(filepath, output_dir, logger):
    """
    Process a GFZ loading file and save it as a pickle using pandas.
    Ensures output is daily with means from higher frequency samples.

    Args:
        filepath (str): Path to the input file
        output_dir (str): Directory to save the pickle file
        logger: Logger object

    Returns:
        dict: Result information including success, resampled status, and error message if any
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
            logger.warning(f"Unknown component in file: {filename}")
            return {
                'success': False,
                'filename': filename,
                'error': "Unknown component"
            }

        # Determine frame (cf or cm)
        if '.cf' in filename:
            frame = 'cf'
        elif '.cm' in filename:
            frame = 'cm'
        else:
            logger.warning(f"Unknown frame in file: {filename}")
            return {
                'success': False,
                'filename': filename,
                'error': "Unknown frame"
            }

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

        # Reorder columns and set index
        df = df[['datetime', 'NS', 'EW', 'R']].set_index('datetime')

        # Check if the data is already daily (by looking at unique days vs total rows)
        # Convert to pandas Series first to use nunique()
        unique_days = pd.Series(df.index.date).nunique()
        is_daily = unique_days == len(df)

        # If data is not daily (3h or other higher frequency), resample to daily mean
        if not is_daily:
            logger.debug(f"Resampling {filename} from {len(df)} points to {unique_days} daily averages")
            # Resample to daily, taking mean and setting time to 12:00 (midday)
            df = df.resample('D').mean()

            # Reset the time to 12:00 for all days (middle of the day)
            new_index = pd.DatetimeIndex([
                pd.Timestamp(date.year, date.month, date.day, 12, 0)
                for date in df.index
            ])

            df.index = new_index

        # Create output filename
        output_filename = f"{charcode}_{component}_{frame}.pkl"
        output_path = os.path.join(output_dir, output_filename)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(df, f)

        # Return success information
        return {
            'success': True,
            'filename': filename,
            'output': output_filename,
            'resampled': not is_daily
        }

    except Exception as e:
        logger.error(f"Error processing {os.path.basename(filepath)}: {str(e)}")
        return {
            'success': False,
            'filename': os.path.basename(filepath),
            'error': str(e)
        }


def main():
    """
    Process all GFZ loading files in the specified directory using multi-threading.
    """
    # Setup logging
    log_dir = "LOGS"
    logger = setup_logging(log_dir)

    start_time = time.time()

    # Directory containing input files
    # input_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = r'D:\from_Kyriakos\non_tidal_loading_station-wise_igs_repro3_extension'
    logger.info(f"Input directory: {input_dir}")

    # Output directory for pickle files
    output_dir = r"EXT\ESMGFZLOADING\CODE"
    logger.info(f"Output directory: {output_dir}")

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all relevant files
    files_to_process = []
    for filename in os.listdir(input_dir):
        # Look for files with the pattern like 00NA-59975M001.lsdm.gfz13.cwsl.cf
        if re.match(r'^\w{4}-.*\.(ntal|ntol|slel|cwsl)\.(cf|cm)$', filename):
            filepath = os.path.join(input_dir, filename)
            files_to_process.append(filepath)

    # files_to_process = files_to_process[:50]

    logger.info(f"Found {len(files_to_process)} files to process")

    # Determine the number of workers (threads)
    # Use max(1, os.cpu_count() - 1) to leave one CPU for system tasks
    num_workers = max(1, os.cpu_count() - 1)
    logger.info(f"Using {num_workers} worker threads")

    # Stats for tracking results
    results = {
        'total': len(files_to_process),
        'success': 0,
        'error': 0,
        'resampled': 0
    }

    # Process files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, filepath, output_dir, logger): filepath
                          for filepath in files_to_process}

        # Create a progress bar
        with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result = future.result()

                    # Update stats based on result
                    if result['success']:
                        results['success'] += 1
                        if result.get('resampled', False):
                            results['resampled'] += 1
                            logger.info(f"Resampled: {result['filename']} -> {result['output']}")
                        else:
                            logger.debug(f"Processed: {result['filename']} -> {result['output']}")
                    else:
                        results['error'] += 1
                        logger.warning(f"Failed: {result['filename']} - {result.get('error', 'Unknown error')}")

                except Exception as exc:
                    results['error'] += 1
                    logger.error(f"{os.path.basename(filepath)} generated an exception: {exc}")

                # Update progress bar
                pbar.update(1)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Log summary
    logger.info(f"--- Processing Summary ---")
    logger.info(f"Total files processed: {results['total']}")
    logger.info(f"Successful: {results['success']}")
    logger.info(f"Failed: {results['error']}")
    logger.info(f"Resampled to daily: {results['resampled']}")
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")

    # Print summary to console
    print(f"\n--- Processing Summary ---")
    print(f"Total files processed: {results['total']}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['error']}")
    print(f"Resampled to daily: {results['resampled']}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"Log file saved to: {log_dir}")


if __name__ == "__main__":
    main()