import os
import pandas as pd
import glob
import re
import concurrent.futures
import logging
import datetime
import time
from pathlib import Path
import numpy as np


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


def add_periodic_signals(df, cpy1_file, cpy2_file, station_code=None, reference_epoch='2015-01-01 00:00:00',
                         logger=None):
    """
    Add periodic signals to dE, dN, dU columns in a dataframe based on coefficients from
    ITRF2020 files for annual (1cpy) and semi-annual (2cpy) periodic signals.

    The function processes each station and SOLN combination separately, applying
    the appropriate coefficients from the files.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'EPOCH' (datetime), 'CODE', 'SOLN', 'FLAG', and dE, dN, dU columns
    cpy1_file : str
        Path to the ITRF2020-1cpy-NEU-CF.dat file (annual signal)
    cpy2_file : str
        Path to the ITRF2020-2cpy-NEU-CF.dat file (semi-annual signal)
    station_code : str, optional
        Four-character station code to filter by. If None, will use the 'CODE' column from the dataframe
    reference_epoch : str, optional
        Reference epoch for the periodic signals, default is '2015-01-01 00:00:00'
    logger : logging.Logger, optional
        Logger object for logging messages

    Returns:
    --------
    pandas.DataFrame
        DataFrame with updated dE, dN, dU columns that include the periodic signals
    """
    # Use provided logger or create a simple one if not provided
    log = logger or logging.getLogger(__name__)

    log.info(f"Adding periodic signals for station {station_code}")

    # Make a copy of the input dataframe to avoid modifying the original
    result_df = df.copy()

    # Ensure EPOCH is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df.index.get_level_values('EPOCH')):
        log.warning("EPOCH is not datetime type, attempting conversion")
        # Create a copy with reset index to manipulate the EPOCH column
        temp_df = result_df.reset_index()
        temp_df['EPOCH'] = pd.to_datetime(temp_df['EPOCH'])
        # Set the index back
        result_df = temp_df.set_index(['EPOCH', 'CODE', 'SOLN', 'FLAG'])
        log.info("Converted EPOCH column to datetime")

    # Convert reference epoch to datetime
    ref_epoch = pd.to_datetime(reference_epoch)
    log.info(f"Using reference epoch: {ref_epoch}")

    # Define periods in days
    period_1cpy = 365.25  # Annual (1 cycle per year)
    period_2cpy = 182.625  # Semi-annual (2 cycles per year)

    # Read coefficient files
    try:
        coef_1cpy = pd.read_csv(cpy1_file, delim_whitespace=True, skiprows=3)
        coef_2cpy = pd.read_csv(cpy2_file, delim_whitespace=True, skiprows=3)
        log.info(f"Successfully read coefficient files: {cpy1_file} and {cpy2_file}")
    except Exception as e:
        log.error(f"Error reading coefficient files: {e}")
        raise

    # Set column names for coefficient files
    coef_1cpy.columns = ['CODE', 'X', 'DOMES', 'SOLN', 'COMP', 'COSX', 'COSX_ERROR', 'SINX', 'SINX_ERROR']
    coef_2cpy.columns = ['CODE', 'X', 'DOMES', 'SOLN', 'COMP', 'COSX', 'COSX_ERROR', 'SINX', 'SINX_ERROR']

    # Create a copy to hold the result
    output_df = pd.DataFrame()

    # Process each SOLN group separately
    unique_solns = list(set(idx[2] for idx in result_df.index))
    log.info(f"Processing {len(unique_solns)} unique SOLN values")

    # Reset index for easier manipulation
    temp_df = result_df.reset_index()

    # Group by SOLN
    for soln, group_df in temp_df.groupby('SOLN'):
        log.info(f"Processing SOLN {soln} with {len(group_df)} records")

        # Get a copy to work with
        soln_df = group_df.copy()

        # Filter coefficients for the specified station and solution number
        soln_coef_1cpy = coef_1cpy[(coef_1cpy['CODE'] == station_code) & (coef_1cpy['SOLN'] == soln)]
        soln_coef_2cpy = coef_2cpy[(coef_2cpy['CODE'] == station_code) & (coef_2cpy['SOLN'] == soln)]

        if soln_coef_1cpy.empty and soln_coef_2cpy.empty:
            log.warning(f"No coefficients found for station code '{station_code}' and SOLN {soln}")
            # Keep original data for this group without changes
            output_df = pd.concat([output_df, soln_df])
            continue

        # Calculate time difference from reference epoch in days
        soln_df['days_since_ref'] = (pd.to_datetime(soln_df['EPOCH']) - ref_epoch).dt.total_seconds() / (24 * 3600)

        # Components mapping
        components = {'N': 'dN', 'E': 'dE', 'U': 'dU'}

        # Add annual signal (1cpy)
        for _, row in soln_coef_1cpy.iterrows():
            component = row['COMP']
            if component in components:
                col = components[component]
                log.debug(f"Adding annual signal for {component} component")

                # Angular frequency (2π/period)
                omega = 2 * np.pi / period_1cpy

                # Calculate periodic signal: A*cos(ωt) + B*sin(ωt)
                cos_term = row['COSX'] * np.cos(omega * soln_df['days_since_ref'])
                sin_term = row['SINX'] * np.sin(omega * soln_df['days_since_ref'])

                # Add to the corresponding displacement column
                soln_df[col] = soln_df[col] + cos_term + sin_term

        # Add semi-annual signal (2cpy)
        for _, row in soln_coef_2cpy.iterrows():
            component = row['COMP']
            if component in components:
                col = components[component]
                log.debug(f"Adding semi-annual signal for {component} component")

                # Angular frequency (2π/period)
                omega = 2 * np.pi / period_2cpy

                # Calculate periodic signal: A*cos(ωt) + B*sin(ωt)
                cos_term = row['COSX'] * np.cos(omega * soln_df['days_since_ref'])
                sin_term = row['SINX'] * np.sin(omega * soln_df['days_since_ref'])

                # Add to the corresponding displacement column
                soln_df[col] = soln_df[col] + cos_term + sin_term

        # Remove the temporary column
        soln_df.drop('days_since_ref', axis=1, inplace=True)

        # Add this processed group to the output
        output_df = pd.concat([output_df, soln_df])
        log.info(f"Completed processing for SOLN {soln}")

    log.info(f"Completed adding periodic signals for station {station_code}")

    # Set the index back
    output_df = output_df.set_index(['EPOCH', 'CODE', 'SOLN', 'FLAG'])

    # Return the combined result
    return output_df


def process_station(station, input_dir, output_dir, solution_name, sampling, add_periodic=False,
                    cpy1_file=None, cpy2_file=None, reference_epoch='2015-01-01 00:00:00'):
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
    add_periodic : bool, optional
        Whether to add periodic signals, default is False
    cpy1_file : str, optional
        Path to the ITRF2020-1cpy-NEU-CF.dat file (annual signal)
    cpy2_file : str, optional
        Path to the ITRF2020-2cpy-NEU-CF.dat file (semi-annual signal)
    reference_epoch : str, optional
        Reference epoch for the periodic signals, default is '2015-01-01 00:00:00'

    Returns:
    --------
    tuple or None
        Tuple containing (station_code, processed_dataframe) or None if processing failed
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
        station_output_dir = os.path.join(output_dir, "DATA", "DISPLACEMENTS", solution_name + f"_{sampling}", 'CODE')
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
        merged_df.columns = [col.upper() if col.lower() in ['epoch', 'code', 'soln', 'flag'] else col for col in
                             merged_df.columns]

        merged_df.rename({'SOLN': 'DOMES'}, axis=1, inplace=True)
        merged_df.loc[:, 'SOLN'] = 1

        # Set the index to [EPOCH, CODE, SOLN, FLAG]
        merged_df = merged_df.set_index(['EPOCH', 'CODE', 'SOLN', 'FLAG'])

        # Add periodic signals if requested
        if add_periodic and cpy1_file and cpy2_file:
            logger.info(f"Adding periodic signals for station {station_code}")
            try:
                merged_df = add_periodic_signals(
                    merged_df,
                    cpy1_file=cpy1_file,
                    cpy2_file=cpy2_file,
                    station_code=station_code,
                    reference_epoch=reference_epoch,
                    logger=logger
                )
                logger.info(f"Successfully added periodic signals for station {station_code}")
            except Exception as e:
                logger.error(f"Error adding periodic signals for station {station_code}: {str(e)}")
                # Don't continue - return None if we can't add periodic signals when requested
                return None

        # Format the displacement values and errors to two decimal places
        for col in ['dN', 'dE', 'dU', 'dN_error', 'dE_error', 'dU_error']:
            merged_df[col] = merged_df[col].round(2)

        # Format the output filename: {Solution}_{4charcode}_{sampling}_DISP.PKL
        output_filename = f"{solution_name}_{station_code}_{sampling}_DISP.PKL"
        output_file = os.path.join(station_output_dir, output_filename)

        # Save to pickle format
        merged_df.to_pickle(output_file)

        logger.info(f"Created: {output_file}")
        return (station_code, merged_df)

    except Exception as e:
        logger.error(f"Error processing station {station}: {str(e)}")
        return None


def process_displacement_files(input_dir, output_dir=None, solution_name=None, sampling="01D",
                               add_periodic=False, cpy1_file=None, cpy2_file=None,
                               reference_epoch='2015-01-01 00:00:00', max_workers=None):
    """
    Process displacement files for multiple stations in a directory using multithreading.

    This function searches for .DE, .DN, and .DH files in the input directory,
    combines them by station, and creates displacement files in the output directory.
    It now also saves data organized by time in addition to by station code.

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
    add_periodic : bool, optional
        Whether to add periodic signals, default is False
    cpy1_file : str, optional
        Path to the ITRF2020-1cpy-NEU-CF.dat file (annual signal)
    cpy2_file : str, optional
        Path to the ITRF2020-2cpy-NEU-CF.dat file (semi-annual signal)
    reference_epoch : str, optional
        Reference epoch for the periodic signals, default is '2015-01-01 00:00:00'
    max_workers : int, optional
        Maximum number of threads to use. If None, will use the default
        based on the number of CPU cores.

    Returns:
    --------
    tuple
        Tuple containing (list of CODE output files, list of TIME output files)
    """
    # Set up logging
    if output_dir is None:
        output_dir = input_dir

    logger = setup_logging(output_dir)

    start_time = time.time()
    logger.info(f"Starting processing with input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Add periodic signals: {add_periodic}")

    if add_periodic:
        if not cpy1_file or not cpy2_file:
            logger.warning("Periodic signal files not provided, disabling periodic signal addition")
            add_periodic = False
        else:
            logger.info(f"Using annual signal file: {cpy1_file}")
            logger.info(f"Using semi-annual signal file: {cpy2_file}")
            logger.info(f"Using reference epoch: {reference_epoch}")

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

    # Verify periodic files exist if add_periodic is True
    if add_periodic:
        if not os.path.exists(cpy1_file) or not os.path.exists(cpy2_file):
            logger.error(f"Cannot find periodic signal files: {cpy1_file} or {cpy2_file}")
            logger.error("Aborting processing as --add_periodic was specified but files are missing")
            return [], []

    # Create the output directory structure for TIME series
    time_output_dir = os.path.join(output_dir, "DATA", "DISPLACEMENTS", solution_name + f"_{sampling}", 'TIME')
    os.makedirs(time_output_dir, exist_ok=True)
    logger.info(f"Created output directory for TIME series: {time_output_dir}")

    # Process stations using multithreading
    station_results = []
    code_output_files = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_station = {
            executor.submit(
                process_station, station, input_dir, output_dir, solution_name, sampling,
                add_periodic, cpy1_file, cpy2_file, reference_epoch
            ): station
            for station in station_names
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_station):
            station = future_to_station[future]
            try:
                result = future.result()
                if result:
                    station_code, merged_df = result
                    station_output_file = os.path.join(
                        output_dir, "DATA", "DISPLACEMENTS",
                        solution_name + f"_{sampling}", 'CODE',
                        f"{solution_name}_{station_code}_{sampling}_DISP.PKL"
                    )
                    code_output_files.append(station_output_file)
                    station_results.append(merged_df)
            except Exception as e:
                logger.error(f"Exception while processing station {station}: {str(e)}")

    # Process time-based output files if we have station results
    time_output_files = []

    if station_results:
        logger.info(f"Processing time-based output files from {len(station_results)} station results")

        # Combine all station results
        try:
            all_stations_df = pd.concat(station_results)

            # Group by time (EPOCH)
            grouped_by_time = all_stations_df.groupby(level='EPOCH')

            for time_idx, time_group in grouped_by_time:
                # Format time for filename
                time_str = pd.to_datetime(time_idx).strftime('%Y%m%d')

                # Create output filename
                time_output_filename = f"{solution_name}_{time_str}_{sampling}_DISP.PKL"
                time_output_path = os.path.join(time_output_dir, time_output_filename)

                # Sort by station code for consistency
                sorted_group = time_group.sort_index(level='CODE')

                # Save to pickle format
                sorted_group.to_pickle(time_output_path)
                time_output_files.append(time_output_path)

                logger.info(f"Created time-based output file: {time_output_path}")
        except Exception as e:
            logger.error(f"Error creating time-based output files: {str(e)}")

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Processing completed. Total time: {duration:.2f} seconds")
    logger.info(f"Successfully processed {len(code_output_files)} out of {len(station_names)} stations")
    logger.info(f"Created {len(time_output_files)} time-based output files")

    return code_output_files, time_output_files


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
    parser.add_argument('--add_periodic', action='store_true', help='Add periodic signals to the data', default=True)
    parser.add_argument('--cpy1_file', help='Path to the ITRF2020-1cpy-ENU-CF.dat file (annual signal)',
                        default='EXT/ITRF_PERIODIC_MODEL/ITRF2020-1cpy-ENU-CF.dat')
    parser.add_argument('--cpy2_file', help='Path to the ITRF2020-2cpy-ENU-CF.dat file (semi-annual signal)',
                        default='EXT/ITRF_PERIODIC_MODEL/ITRF2020-2cpy-ENU-CF.dat')
    parser.add_argument('--reference_epoch', help='Reference epoch for the periodic signals',
                        default='2015-01-01 00:00:00')

    args = parser.parse_args()

    code_files, time_files = process_displacement_files(
        args.input_dir,
        args.output_dir,
        args.solution,
        args.sampling,
        args.add_periodic,
        args.cpy1_file if args.add_periodic else None,
        args.cpy2_file if args.add_periodic else None,
        args.reference_epoch,
        args.threads
    )

    # The detailed summary is now in the log file
    logger = logging.getLogger()
    logger.info(f"Processed {len(code_files)} station(s) with CODE-based output files.")
    logger.info(f"Created {len(time_files)} TIME-based output files.")

    if code_files:
        try:
            sample_df = pd.read_pickle(code_files[0])
            logger.info(f"Sample CODE file: {code_files[0]}")
            logger.info(f"Sample shape: {sample_df.shape}")
            logger.info(f"Sample index names: {sample_df.index.names}")
            logger.info(f"Sample column names: {sample_df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error reading sample CODE file: {str(e)}")

    if time_files:
        try:
            sample_time_df = pd.read_pickle(time_files[0])
            logger.info(f"Sample TIME file: {time_files[0]}")
            logger.info(f"Sample shape: {sample_time_df.shape}")
            logger.info(f"Sample index names: {sample_time_df.index.names}")
            logger.info(f"Sample column names: {sample_time_df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error reading sample TIME file: {str(e)}")


if __name__ == "__main__":
    main()