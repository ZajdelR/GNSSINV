import logging
import os
import gfzload2snx_tools as gfztl
from geodezyx import conv
import pandas as pd
import numpy as np
from glob import glob
import concurrent.futures
from functools import partial
import time
from datetime import datetime

def transform_station_coordinates(df, valuecolumn, errorcolumn):
    """
    Transform coordinate data from STAX/STAY/STAZ format to a unified table with X, Y, Z components

    Parameters:
    df (pandas.DataFrame): DataFrame with coordinate data

    Returns:
    pandas.DataFrame: Transformed DataFrame with columns epoch, CODE, SOLN, S, X, Y, Z, X_error, Y_error, Z_error
    """
    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Extract coordinate type from _TYPE_ column
    df_copy['coord_type'] = df_copy['TYPE'].str.replace('STA', '')

    # Get unique station codes
    station_codes = df_copy['CODE'].unique()

    # Prepare for vectorized operations
    prefix = f"{valuecolumn[:3]}_"

    # Process each station - create a list of dictionaries for each station
    rows = []
    for code in station_codes:
        station_data = df_copy[df_copy['CODE'] == code]

        # Get component data using boolean indexing (faster than multiple .iloc calls)
        x_mask = station_data['coord_type'] == 'X'
        y_mask = station_data['coord_type'] == 'Y'
        z_mask = station_data['coord_type'] == 'Z'

        # Get rows for each component
        x_data = station_data[x_mask]
        y_data = station_data[y_mask]
        z_data = station_data[z_mask]

        # Skip if no components found
        if len(x_data) + len(y_data) + len(z_data) == 0:
            continue

        # Use first available component for reference values
        reference_row = x_data.iloc[0] if not x_data.empty else (
            y_data.iloc[0] if not y_data.empty else z_data.iloc[0])

        # Create new row
        new_row = {
            'EPOCH': reference_row['REFEPOCH'],
            'CODE': code,
            'SOLN': reference_row['SOLN'],
            'S': reference_row['S'],
            f'{prefix}X': x_data[valuecolumn].iloc[0] if not x_data.empty else np.nan,
            f'{prefix}Y': y_data[valuecolumn].iloc[0] if not y_data.empty else np.nan,
            f'{prefix}Z': z_data[valuecolumn].iloc[0] if not z_data.empty else np.nan,
            f'{prefix}X_error': x_data[errorcolumn].iloc[0] if not x_data.empty else np.nan,
            f'{prefix}Y_error': y_data[errorcolumn].iloc[0] if not y_data.empty else np.nan,
            f'{prefix}Z_error': z_data[errorcolumn].iloc[0] if not z_data.empty else np.nan
        }

        rows.append(new_row)

    # Create DataFrame from rows (more efficient than appending to DataFrame)
    return pd.DataFrame(rows)


def process_sinex_file(snx_path, solution, outdir):
    """Process a single SINEX file and save the results."""
    try:
        basename = os.path.basename(snx_path)
        print(f'Processing: {basename}')

        basename_pkl = basename.replace('.SNX', '.PKL')
        basename_apr = basename_pkl.replace('CRD', 'APR')
        basename_est = basename_pkl.replace('CRD', 'EST')

        # Read SINEX data - this is likely IO bound
        dfapr = gfztl.read_sinex_versatile(snx_path, "SOLUTION/APRIORI")
        dfapr.columns = [x.replace('_', '') for x in dfapr.columns]
        dfapr = dfapr[dfapr['TYPE'].str.startswith('STA')]

        dfest = gfztl.read_sinex_versatile(snx_path, "SOLUTION/ESTIMATE")
        dfest.columns = [x.replace('_', '') for x in dfest.columns]
        dfest = dfest[dfest['TYPE'].str.startswith('STA')]

        # Transform coordinates - this is CPU bound
        dfapr2 = transform_station_coordinates(dfapr, 'APRIORIVALUE', 'STDDEV').set_index(
            ['EPOCH', 'CODE', 'SOLN', 'S'])
        dfest2 = transform_station_coordinates(dfest, 'ESTIMATEDVALUE', 'STDDEV').set_index(
            ['EPOCH', 'CODE', 'SOLN', 'S'])

        # Save results - IO bound
        dfapr2.to_pickle(os.path.join(outdir, basename_apr))
        dfest2.to_pickle(os.path.join(outdir, basename_est))

        return basename, True
    except Exception as e:
        return basename, f"Error: {str(e)}"

def filter_files_by_date(all_files, start_date, end_date):
    filtered_files = []
    for filename in all_files:
        # Extract the date portion from the filename (after the first underscore)
        file = os.path.basename(filename)

        date_part = file.split('_')[1]

        # Extract year and day of year
        year = int(date_part[:4])
        doy = int(date_part[4:7])

        # Convert to datetime
        file_date = datetime(year, 1, 1) + pd.Timedelta(days=doy - 1)

        # Check if the date is within our range
        if start_date <= file_date < end_date:
            filtered_files.append(filename)
    return filtered_files
def main():
    # Start timing
    start_time = time.time()

    # Configuration
    solution = 'IGS1R03SNX'
    # input_dir = r'C:\Users\rados\Documents\VM_SHARED\SNX_DATA'
    input_dir = r'DATA\SNX_ORIGINAL'
    output_dir = r'DATA\SNX_OUT_PKL'

    # Create output directory
    outdir = os.path.join(output_dir, solution)
    os.makedirs(outdir, exist_ok=True)

    # Get list of files to process
    files = glob(os.path.join(input_dir, solution, '*.SNX'))

    # Define start and end dates
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2023, 1, 1)

    files = filter_files_by_date(files, start_date, end_date)

    # Display info
    print(f"Found {len(files)} SINEX files to process")

    # Create partial function with fixed parameters
    process_file = partial(process_sinex_file, solution=solution, outdir=outdir)

    # Use a ProcessPoolExecutor for CPU-bound operations
    # The max_workers=None will use the number of processors on the machine
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        # Submit all files for processing
        future_to_file = {executor.submit(process_file, file): file for file in files}

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            filename, result = future.result()
            if result is True:
                print(f"Completed: {filename}")
            else:
                print(f"Failed: {filename} - {result}")

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Average time per file: {elapsed_time / len(files):.2f} seconds")

if __name__ == "__main__":
    main()