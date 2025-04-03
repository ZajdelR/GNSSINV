import os
import pandas as pd
import glob
import re
import concurrent.futures
import logging
import datetime
import time
from pathlib import Path


def setup_logging(output_dir):
    """
    Set up logging to both console and file.

    Parameters:
    -----------
    output_dir : str
        Base directory where logs will be saved

    Returns:
    --------
    logger : logging.Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_dir, "LOGS")
    os.makedirs(logs_dir, exist_ok=True)

    # Get script name without extension
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # Create timestamped log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"LOG_{script_name}_{timestamp}.txt"
    log_filepath = os.path.join(logs_dir, log_filename)

    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler for saving to file
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler for displaying in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging started. Log file: {log_filepath}")

    return logger


def process_station(station, input_dir, output_dir, solution_name, sampling):
    """
    Process a single station's displacement files.

    Parameters:
    -----------
    station : str
        Station identifier
    input_dir : str
        Path to directory containing the displacement files
    output_dir : str
        Path to directory where output files will be saved
    solution_name : str
        Name of the solution for organizing output files
    sampling : str
        Sampling rate for the data

    Returns:
    --------
    str or None
        Path to the created output file, or None if processing failed
    """
    try:
        logger = logging.getLogger()
        logger.info(f"Processing station: {station}")

        # Extract the CODE (station code) from the station identifier
        station_parts = station.split('_')
        if len(station_parts) >= 1:
            station_code = station_parts[0]
            # Ensure station code is exactly 4 characters
            station_code = station_code[:4]
            if len(station_code) < 4:
                logger.warning(f"Station code {station_code} is less than 4 characters, skipping.")
                return None
        else:
            station_code = station[:4]  # Use first 4 chars if no underscore
            if len(station_code) < 4:
                logger.warning(f"Station code {station_code} is less than 4 characters, skipping.")
                return None

        # Create the output directory structure: DATA\DISPLACEMENTS\{solution}\CODE
        station_output_dir = os.path.join(output_dir, "DATA", "DISPLACEMENTS", solution_name, 'CODE')
        os.makedirs(station_output_dir, exist_ok=True)

        # File paths for this station
        de_file = os.path.join(input_dir, f"{station}.DE")
        dn_file = os.path.join(input_dir, f"{station}.DN")
        dh_file = os.path.join(input_dir, f"{station}.DH")

        # Check if all required files exist
        if not (os.path.exists(de_file) and os.path.exists(dn_file) and os.path.exists(dh_file)):
            logger.warning(f"Missing files for station {station}, skipping.")
            return None

        # Read files using pandas
        # Define column names based on the file format
        cols = ['epoch', 'value', 'std', 'code', 'soln', 'flag']

        # Read each file with whitespace delimiter
        de_data = pd.read_csv(de_file, sep=r'\s+', header=None, names=cols)
        dn_data = pd.read_csv(dn_file, sep=r'\s+', header=None, names=cols)
        dh_data = pd.read_csv(dh_file, sep=r'\s+', header=None, names=cols)

        # Rename the value and std columns to represent the components and their errors
        de_data = de_data.rename(columns={'value': 'dE', 'std': 'dE_error'})
        dn_data = dn_data.rename(columns={'value': 'dN', 'std': 'dN_error'})
        dh_data = dh_data.rename(columns={'value': 'dU', 'std': 'dU_error'})

        # Merge the dataframes on the epoch
        merged_df = pd.merge(pd.merge(de_data[['epoch', 'dE', 'dE_error', 'code', 'soln', 'flag']],
                                      dn_data[['epoch', 'dN', 'dN_error']],
                                      on='epoch'),
                             dh_data[['epoch', 'dU', 'dU_error']],
                             on='epoch')

        # Convert epoch to datetime (converting decimal years to datetime)
        def decimal_year_to_datetime(decimal_year):
            year = int(decimal_year)
            fraction = decimal_year - year
            # Calculate the number of days in the year (accounting for leap years)
            days_in_year = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365
            day_of_year = int(fraction * days_in_year)
            # Convert to datetime
            return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=day_of_year)

        # Apply the conversion function to the epoch column
        merged_df['epoch'] = merged_df['epoch'].apply(decimal_year_to_datetime)

        # Convert column names to uppercase
        merged_df.columns = [col.upper() if col in ['epoch', 'code', 'soln', 'flag'] else col for col in
                             merged_df.columns]

        # Set the index to [EPOCH, CODE, SOLN, FLAG]
        merged_df = merged_df.set_index(['EPOCH', 'CODE', 'SOLN', 'FLAG'])

        # Format the output filename: {Solution}_{4charcode}_{sampling}_DISP.PKL
        output_filename = f"{solution_name}_{station_code}_{sampling}_DISP.PKL"
        output_file = os.path.join(station_output_dir, output_filename)

        # Format the displacement values and errors to two decimal places
        for col in ['dN', 'dE', 'dU', 'dN_error', 'dE_error', 'dU_error']:
            merged_df[col] = merged_df[col].round(2)

        # Save to pickle format
        merged_df.to_pickle(output_file)

        logger.info(f"Created: {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error processing station {station}: {str(e)}")
        return None


def process_displacement_files(input_dir, output_dir=None, solution_name=None, sampling="01D", max_workers=None):
    """
    Process displacement files for multiple stations in a directory using multithreading.

    This function searches for .DE, .DN, and .DH files in the input directory,
    combines them by station, and creates displacement files in the output directory.

    Parameters:
    -----------
    input_dir : str
        Path to directory containing the displacement files
    output_dir : str, optional
        Path to directory where output files will be saved
        If None, files will be saved in the input directory
    solution_name : str, optional
        Name of the solution for organizing output files
        If None, will use the last part of the input_dir
    sampling : str, optional
        Sampling rate for the data, default is "01D" (1 day)
    max_workers : int, optional
        Maximum number of threads to use. If None, will use the default
        based on the number of CPU cores.

    Returns:
    --------
    list
        List of paths to created output files
    """
    # Set up logging
    if output_dir is None:
        output_dir = input_dir

    logger = setup_logging(output_dir)

    start_time = time.time()
    logger.info(f"Starting processing with input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # If solution_name is not provided, extract it from the input directory
    if solution_name is None:
        solution_name = os.path.basename(os.path.normpath(input_dir))
    logger.info(f"Solution name: {solution_name}")

    # Find all station codes by looking at the filenames
    # Format is typically XXXX_YYYYYYY.DE/DN/DH
    de_files = glob.glob(os.path.join(input_dir, "*.DE"))
    logger.info(f"Found {len(de_files)} DE files")

    # Extract station names from file names
    station_pattern = re.compile(r"([A-Z0-9]+_[A-Z0-9]+)\.DE")
    station_names = []

    for file in de_files:
        base_name = os.path.basename(file)
        match = station_pattern.match(base_name)
        if match:
            station_names.append(match.group(1))

    logger.info(f"Identified {len(station_names)} stations to process")

    # Process stations using multithreading
    output_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_station = {
            executor.submit(process_station, station, input_dir, output_dir, solution_name, sampling): station
            for station in station_names
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_station):
            station = future_to_station[future]
            try:
                output_file = future.result()
                if output_file:
                    output_files.append(output_file)
            except Exception as e:
                logger.error(f"Exception while processing station {station}: {str(e)}")

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Processing completed. Total time: {duration:.2f} seconds")
    logger.info(f"Successfully processed {len(output_files)} out of {len(station_names)} stations")

    return output_files


def main():
    """
    Example usage of the process_displacement_files function.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Process displacement files for multiple stations.')
    parser.add_argument('--input_dir', help='Directory containing displacement files',
                        default=r'D:\from_Kyriakos\ITRF2020-IGS-RES')
    parser.add_argument('--output_dir', help='Base directory for output files', default=r'.')
    parser.add_argument('--solution', help='Solution name (default: derived from input directory)',
                        default=None)
    parser.add_argument('--sampling', help='Sampling rate (default: 01D)', default='01D')
    parser.add_argument('--threads', type=int, help='Number of threads to use (default: CPU count)', default=None)

    args = parser.parse_args()

    output_files = process_displacement_files(
        args.input_dir,
        args.output_dir,
        args.solution,
        args.sampling,
        args.threads
    )

    # The detailed summary is now in the log file
    logger = logging.getLogger()
    logger.info(f"Processed {len(output_files)} station(s).")

    if output_files:
        try:
            sample_df = pd.read_pickle(output_files[0])
            logger.info(f"Sample file: {output_files[0]}")
            logger.info(f"Sample shape: {sample_df.shape}")
            logger.info(f"Sample index names: {sample_df.index.names}")
            logger.info(f"Sample column names: {sample_df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error reading sample file: {str(e)}")


if __name__ == "__main__":
    main()