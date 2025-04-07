import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
import glob
from pathlib import Path
import re


def grid_to_dataframe(file_path, resolution=1):
    """
    Convert a NetCDF grid file to a pandas DataFrame with improved efficiency.
    Resamples the grid from 0.5x0.5 to 1x1 degree resolution.

    Parameters:
    -----------
    file_path : str
        Path to the NetCDF file

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the grid data with columns for displacements at 1x1 degree resolution
    """
    # Extract model name from filename
    filename = os.path.basename(file_path)
    match = re.search(r'ESMGFZ(.*?)_', filename)
    model = match.group(1) if match else 'UNKNOWN'

    print(f"Processing {filename}...")

    # Open the NetCDF file with xarray, which is memory efficient
    try:
        ds = xr.open_dataset(file_path)

        print(
            f"File dimensions: time={ds.dims.get('time', 0)}, lat={ds.dims.get('lat', 0)}, lon={ds.dims.get('lon', 0)}")

        # Resample from 0.5x0.5 to 1x1 grid
        # Creating new coordinate arrays for 1 degree resolution
        new_lat = np.arange(np.floor(ds.lat.min()), np.ceil(ds.lat.max()), resolution)
        new_lon = np.arange(np.floor(ds.lon.min()), np.ceil(ds.lon.max()), resolution)

        # Using coarsen or resample for downsampling
        if 'lat' in ds.dims and 'lon' in ds.dims:
            # Check if current resolution is already 1 degree
            lat_resolution = abs(ds.lat[1] - ds.lat[0])
            lon_resolution = abs(ds.lon[1] - ds.lon[0])

            if abs(lat_resolution - 0.5) < 0.01 and abs(lon_resolution - 0.5) < 0.01:
                print(f"Resampling from 0.5x0.5 to {resolution:.1f}x{resolution:.1f} grid...")
                # Resampling to 1x1 grid using nearest or averaging method
                ds = ds.interp(lat=new_lat, lon=new_lon, method='nearest')
            else:
                print(f"Current resolution is approximately {lat_resolution}x{lon_resolution} degrees")
                # If not 0.5x0.5, still resample to 1x1 to ensure consistent output
                ds = ds.interp(lat=new_lat, lon=new_lon, method='nearest')

        # Create column names based on the model
        new_column_names = {}
        for var in ds.data_vars:
            if var == 'duV':
                new_column_names[var] = 'dU'
            elif var == 'duNS':
                new_column_names[var] = 'dN'
            elif var == 'duEW':
                new_column_names[var] = 'dE'

        # Rename variables before converting to DataFrame
        ds = ds.rename(new_column_names)

        # Convert to DataFrame and reset index to get lat, lon, time as columns
        df = ds.to_dataframe().reset_index()

        # Keep only the necessary columns to reduce memory usage
        displacement_cols = [col for col in df.columns if col.startswith('d')]
        df = df[['time', 'lat', 'lon'] + displacement_cols]
        df = df.dropna(subset=displacement_cols,how='any')
        df.loc[:,displacement_cols] *= 1e3
        print(f"DataFrame created with {len(df):,} rows and {len(df.columns)} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

        return df

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None


def convert_nc_to_daily_df(source_dir, model, model_outputname, year, frame, output_resolution, output_dir=None):
    """
    Convert yearly NC grid files into daily dataframes with time, Latitude, Longitude, DN, DE, and DU.
    The grid resolution is dajusted to the needs.

    Parameters:
    -----------
    source_dir : str
        Directory containing the NC files
    model : str
        Name of the model (used for file naming)
    year : int or str
        Year of the data
    frame : str
        Reference frame (e.g., 'CF', 'CM')
    output_dir : str, optional
        Directory to save the output files. If None, files are saved in source_dir/processed/

    Returns:
    --------
    list
        List of paths to the created dataframe files
    """
    # Ensure year is a string
    year = str(year)

    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(source_dir, 'processed')

    os.makedirs(output_dir, exist_ok=True)

    # Find all NC files for the given model and year
    file_pattern = f"{model}*{frame}*{year}*.nc"
    nc_files = glob.glob(os.path.join(source_dir, file_pattern))

    if len(nc_files) == 0:
        raise FileNotFoundError(f"No NC files found matching pattern {file_pattern} in {source_dir}")

    output_files = []

    for nc_file in nc_files:
        # Convert the entire file to a DataFrame using the grid_to_dataframe function
        # which now resamples to 1x1 grid
        df_full = grid_to_dataframe(nc_file, resolution=output_resolution)

        if df_full is None:
            print(f"Skipping file {nc_file} due to processing error")
            continue

        # Get unique times in the dataframe
        monthly_groups = df_full.groupby(pd.Grouper(key='time', freq='1D'))

        for month, df_time in monthly_groups:
            # Standardize column names
            df_time = df_time.rename(columns={
                'time': 'EPOCH',
                'lat': 'Latitude',
                'lon': 'Longitude',
            })

            year_month = datetime.strftime(month, "%Y%m%d")
            # Create output filename
            output_file = f"{model.split('_')[0]}_{year_month}_{model_outputname[7:]}_DISP.PKL"  # Added 1x1 to filename to indicate resolution
            output_path = os.path.join(output_dir, output_file)

            # Save to pickle
            df_time.to_pickle(output_path)
            output_files.append(output_path)

            print(f"Created daily dataframe for {year_month} with {output_resolution:.1f} deg grid: {output_path}")

    return output_files


# Example usage (directly in the script)
if __name__ == "__main__":
    # Configuration
    SOURCE_DIR = r"D:\from_Kyriakos\PROCESSING\ESMGFZLOAD_GRID\LSDM"  # Replace with your source directory
    MODEL = "ESMGFZ_HYDL"  # Replace with your model name
    FRAME = "CF"  # Replace with desired reference frame
    # MODEL_OUTPUTNAME = f'{MODEL.split('_')[0]}_{MODEL.split('_')[1][0]}_{FRAME.lower()}'
    MODEL_OUTPUTNAME = f'{MODEL.split('_')[0]}_LSDM_{FRAME.lower()}'
    OUTPUT_RESOLUTION = 7.5
    OUTPUT_DIR = f"DATA/DISPLACEMENTS/{MODEL_OUTPUTNAME}_GRIDS/TIME/"  # Replace with your output directory

    for YEAR in range(2015, 2023):
        # Call the function
        output_files = convert_nc_to_daily_df(
            source_dir=SOURCE_DIR,
            model=MODEL,
            model_outputname=MODEL_OUTPUTNAME,
            year=YEAR,
            frame=FRAME,
            output_resolution=OUTPUT_RESOLUTION,
            output_dir=OUTPUT_DIR
        )
        # break

    print(f"Conversion complete. Created {len(output_files)} daily files.")
