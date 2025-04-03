import os
import logging
from glob import glob
from datetime import datetime
import matplotlib
import pandas as pd
import numpy as np

matplotlib.use('TkAgg')


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

def setup_logging(script_name):
    """
    Set up logging to file and console.

    Parameters:
    -----------
    script_name : str
        Name of the script for the log file naming

    Returns:
    --------
    logger : logging.Logger
        Configured logger object
    """
    # Create logs directory if it doesn't exist
    log_dir = "LOGS"
    os.makedirs(log_dir, exist_ok=True)

    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"LOG_{script_name}_{timestamp}.log")

    # Configure logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging setup complete. Log file: {log_file}")
    return logger


def add_periodic_signals(df, cpy1_file, cpy2_file, station_code=None, reference_epoch='2015-01-01 00:00', logger=None):
    """
    Add periodic signals to dX, dY, and dZ columns in a dataframe based on coefficients from
    ITRF2020 files for annual (1cpy) and semi-annual (2cpy) periodic signals.

    The function processes each station and SOLN combination separately, applying
    the appropriate coefficients from the files.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'EPOCH' (datetime), 'CODE', 'SOLN', and dX, dY, dZ columns
    cpy1_file : str
        Path to the ITRF2020-1cpy-XYZ-CF.dat file (annual signal)
    cpy2_file : str
        Path to the ITRF2020-2cpy-XYZ-CF.dat file (semi-annual signal)
    station_code : str, optional
        Four-character station code to filter by. If None, will use the 'CODE' column from the dataframe
    reference_epoch : str, optional
        Reference epoch for the periodic signals, default is '2015-01-01 00:00'
    logger : logging.Logger, optional
        Logger object for logging messages

    Returns:
    --------
    pandas.DataFrame
        DataFrame with updated dX, dY, dZ columns that include the periodic signals
    """
    # Use provided logger or create a simple one if not provided
    log = logger or logging.getLogger(__name__)

    log.info(f"Adding periodic signals for station {station_code}")

    # Make a copy of the input dataframe to avoid modifying the original
    result_df = df.copy()

    # Ensure EPOCH is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df['EPOCH']):
        result_df['EPOCH'] = pd.to_datetime(result_df['EPOCH'])
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

    coef_1cpy.columns = ['CODE', 'X', 'DOMES', 'SOLN', 'COMP', 'COSX', 'COSX_ERROR', 'SINX', 'SINX_ERROR']
    coef_2cpy.columns = ['CODE', 'X', 'DOMES', 'SOLN', 'COMP', 'COSX', 'COSX_ERROR', 'SINX', 'SINX_ERROR']

    # Ensure we have SOLN column in the dataframe
    if 'SOLN' not in result_df.columns:
        log.error("Input dataframe must have a 'SOLN' column to match with coefficient files")
        raise ValueError("Input dataframe must have a 'SOLN' column to match with coefficient files")

    # Create a copy to hold the result
    output_df = pd.DataFrame()

    # Process each SOLN group separately
    log.info(f"Processing {len(result_df['SOLN'].unique())} unique SOLN values")
    for soln, group_df in result_df.groupby('SOLN'):
        log.info(f"Processing SOLN {soln} with {len(group_df)} records")

        # Get a copy to work with
        temp_df = group_df.copy()

        # Filter coefficients for the specified station and solution number
        soln_coef_1cpy = coef_1cpy[(coef_1cpy['CODE'] == station_code) & (coef_1cpy['SOLN'] == soln)]
        soln_coef_2cpy = coef_2cpy[(coef_2cpy['CODE'] == station_code) & (coef_2cpy['SOLN'] == soln)]

        if soln_coef_1cpy.empty and soln_coef_2cpy.empty:
            log.warning(f"No coefficients found for station code '{station_code}' and SOLN {soln}")
            # Keep original data for this group without changes
            output_df = pd.concat([output_df, temp_df])
            continue

        # Calculate time difference from reference epoch in days
        temp_df['days_since_ref'] = (temp_df['EPOCH'] - ref_epoch).dt.total_seconds() / (24 * 3600)

        # Components mapping
        components = {'X': 'dX', 'Y': 'dY', 'Z': 'dZ'}

        # Add annual signal (1cpy)
        for _, row in soln_coef_1cpy.iterrows():
            component = row['COMP']
            if component in components:
                col = components[component]
                log.debug(f"Adding annual signal for {component} component")

                # Angular frequency (2π/period)
                omega = 2 * np.pi / period_1cpy

                # Calculate periodic signal: A*cos(ωt) + B*sin(ωt)
                cos_term = row['COSX'] * np.cos(omega * temp_df['days_since_ref'])
                sin_term = row['SINX'] * np.sin(omega * temp_df['days_since_ref'])

                # Add to the corresponding displacement column
                temp_df[col] = temp_df[col] + cos_term + sin_term

        # Add semi-annual signal (2cpy)
        for _, row in soln_coef_2cpy.iterrows():
            component = row['COMP']
            if component in components:
                col = components[component]
                log.debug(f"Adding semi-annual signal for {component} component")

                # Angular frequency (2π/period)
                omega = 2 * np.pi / period_2cpy

                # Calculate periodic signal: A*cos(ωt) + B*sin(ωt)
                cos_term = row['COSX'] * np.cos(omega * temp_df['days_since_ref'])
                sin_term = row['SINX'] * np.sin(omega * temp_df['days_since_ref'])

                # Add to the corresponding displacement column
                temp_df[col] = temp_df[col] + cos_term + sin_term

        # Remove the temporary column
        temp_df.drop('days_since_ref', axis=1, inplace=True)

        # Add this processed group to the output
        output_df = pd.concat([output_df, temp_df])
        log.info(f"Completed processing for SOLN {soln}")

    log.info(f"Completed adding periodic signals for station {station_code}")
    # Return the combined result with index reset
    return output_df.reset_index(drop=True)


def ecef_to_topo(df, latitude, longitude, logger=None):
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) displacements to
    local topocentric coordinates (North, East, Up).

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing dX, dY, dZ columns with ECEF displacements
    latitude : float
        Latitude of the station in degrees
    longitude : float
        Longitude of the station in degrees
    logger : logging.Logger, optional
        Logger object for logging messages

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional dN, dE, dU columns
    """
    # Use provided logger or create a simple one if not provided
    log = logger or logging.getLogger(__name__)

    log.info(f"Converting ECEF to topocentric coordinates for lat={latitude}, lon={longitude}")

    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()

    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)

    # ECEF to topocentric transformation matrix
    # This rotation matrix converts from ECEF to local NEU coordinates
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    log.info(f"Processing {len(result_df)} records for ECEF to NEU conversion")

    # For each row in the DataFrame, compute the transformation
    dN = []
    dE = []
    dU = []

    for _, row in result_df.iterrows():
        dx = row['dX']
        dy = row['dY']
        dz = row['dZ']

        # Apply rotation matrix
        dn = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        de = -sin_lon * dx + cos_lon * dy
        du = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        dN.append(dn)
        dE.append(de)
        dU.append(du)

    # Add new columns to the DataFrame
    result_df['dN'] = dN  # North displacement
    result_df['dE'] = dE  # East displacement
    result_df['dU'] = dU  # Up displacement

    log.info("ECEF to topocentric conversion completed")
    return result_df


def process_data(solution, sampling, start_date, end_date, add_periodic=True, log_level=logging.INFO):
    """
    Main function to process the data.

    Parameters:
    -----------
    solution : str
        Solution identifier (e.g., 'IGS1R03SNX')
    sampling : str
        Sampling identifier (e.g., '01D')
    start_date : datetime
        Start date for filtering files
    end_date : datetime
        End date for filtering files
    add_periodic : bool, optional
        Whether to add periodic signals, default is True
    log_level : int, optional
        Logging level, default is logging.INFO

    Returns:
    --------
    pd.DataFrame
        Combined results DataFrame
    """
    # Setup logging
    logger = setup_logging(f"process_data_{solution}_{sampling}")
    logger.setLevel(log_level)

    logger.info(f"Starting data processing for solution {solution} with sampling {sampling}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Add periodic signals: {add_periodic}")

    # Create output directories
    out_dir_code = rf'DATA/DISPLACEMENTS/{solution}_{sampling}/CODE/'
    out_dir_time = rf'DATA/DISPLACEMENTS/{solution}_{sampling}/TIME/'
    os.makedirs(out_dir_code, exist_ok=True)
    os.makedirs(out_dir_time, exist_ok=True)
    logger.info(f"Created output directories: {out_dir_code} and {out_dir_time}")

    # Find input files
    files_apr = glob(f'DATA/SNX_OUT_PKL/{solution}/{solution}*{sampling}*APR.PKL')
    files_est = glob(f'DATA/SNX_OUT_PKL/{solution}/{solution}*{sampling}*EST.PKL')
    logger.info(f"Found {len(files_apr)} APR files and {len(files_est)} EST files")

    # Filter files by date
    files_apr = filter_files_by_date(files_apr, start_date, end_date)
    files_est = filter_files_by_date(files_est, start_date, end_date)
    logger.info(f"After date filtering: {len(files_apr)} APR files and {len(files_est)} EST files")

    # Read and concat files
    try:
        apr = pd.concat([pd.read_pickle(x) for x in files_apr], axis=0)
        est = pd.concat([pd.read_pickle(x) for x in files_est], axis=0)
        logger.info(f"Successfully read and concatenated input files")
    except Exception as e:
        logger.error(f"Error reading input files: {e}")
        raise

    # Combine data
    final = pd.concat([apr, est], axis=1)
    final.loc[:, ['dX', 'dY', 'dZ']] = (final[['EST_X', 'EST_Y', 'EST_Z']].values -
                                        final[['APR_X', 'APR_Y', 'APR_Z']].values) * 1e3
    logger.info(f"Created combined dataframe with {len(final)} records")

    # Load lat/lon data
    try:
        latlon = pd.read_pickle(r'EXT/PROCESSINS_SUPPLEMENTS/ALL_STATIONS_LATLON.pkl')
        logger.info(f"Successfully loaded station lat/lon data with {len(latlon)} stations")
    except Exception as e:
        logger.error(f"Error loading station lat/lon data: {e}")
        raise

    # Process by station
    dfsta = final.groupby('CODE')
    logger.info(f"Processing {len(dfsta)} unique stations")

    results = []
    for sta, df in dfsta:
        logger.info(f"Processing station {sta} with {len(df)} records")

        # Add periodic signals if requested
        if add_periodic:
            try:
                df_processed = add_periodic_signals(
                    df.reset_index(),
                    'EXT/ITRF_PERIODIC_MODEL/ITRF2020-1cpy-XYZ-CF.dat',
                    'EXT/ITRF_PERIODIC_MODEL/ITRF2020-2cpy-XYZ-CF.dat',
                    station_code=sta,
                    logger=logger
                )
            except Exception as e:
                logger.error(f"Error adding periodic signals for station {sta}: {e}")
                df_processed = df.reset_index()  # Use original data if error occurs
        else:
            logger.info(f"Skipping periodic signal addition for station {sta} as requested")
            df_processed = df.reset_index()

        # Get station lat/lon
        try:
            stalatlon = latlon.loc[sta]
            logger.info(f"Station {sta} coordinates: Lat={stalatlon['Latitude']}, Lon={stalatlon['Longitude']}")
        except KeyError:
            logger.error(f"No lat/lon data found for station {sta}")
            continue

        # Convert to topocentric coordinates
        try:
            df_with_topo = ecef_to_topo(
                df_processed,
                latitude=stalatlon['Latitude'],
                longitude=stalatlon['Longitude'],
                logger=logger
            )
        except Exception as e:
            logger.error(f"Error converting to topocentric coordinates for station {sta}: {e}")
            continue

        # Save results
        try:
            result = df_with_topo.set_index(['EPOCH', 'CODE', 'SOLN', 'S'])[['dX', 'dY', 'dZ', 'dN', 'dE', 'dU']]
            output_path = os.path.join(out_dir_code, f'{solution}_{sta}_{sampling}_DISP.PKL')
            result.to_pickle(output_path)
            logger.info(f"Saved results for station {sta} to {output_path}")
            results.append(result)
        except Exception as e:
            logger.error(f"Error saving results for station {sta}: {e}")

    # Combine all results
    try:
        if results:
            results_df = pd.concat(results, axis=0)
            logger.info(f"Combined results dataframe created with {len(results_df)} records")

            # Group by time and save
            dftime = results_df.groupby('EPOCH')
            logger.info(f"Saving results by time for {len(dftime)} unique epochs")

            for time, df in dftime:
                str_time = time.strftime('%Y%m%d')
                output_path = os.path.join(out_dir_time, f'{solution}_{str_time}_{sampling}_DISP.PKL')
                df.sort_index(level='CODE').to_pickle(output_path)
                logger.info(f"Saved time-based results for {str_time} to {output_path}")

            logger.info("Data processing completed successfully")
            return results_df
        else:
            logger.warning("No results to combine")
            return None
    except Exception as e:
        logger.error(f"Error in final data processing: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    solution = 'IGS1R03SNX'  # or 'TUG0R03FIN'
    sampling = '01D'

    # Define start and end dates
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2024, 1, 1)

    # Process data with optional parameters
    results = process_data(
        solution=solution,
        sampling=sampling,
        start_date=start_date,
        end_date=end_date,
        add_periodic=True,  # Set to False to skip periodic signal addition
        log_level=logging.INFO  # Use logging.DEBUG for more verbose output
    )