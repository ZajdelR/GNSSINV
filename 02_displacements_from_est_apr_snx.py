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


def process_data(solution, sampling, start_date, end_date, add_periodic=True, log_level=logging.INFO):
    """
    Main function to process the data with improved transformation order.

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
    out_dir_code = rf'DATA/DISPLACEMENTS/{solution}_{sampling}_2/CODE/'
    out_dir_time = rf'DATA/DISPLACEMENTS/{solution}_{sampling}_2/TIME/'
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
    final.loc[:, ['dX', 'dY', 'dZ']] = -(final[['EST_X', 'EST_Y', 'EST_Z']].values -
                                        final[['APR_X', 'APR_Y', 'APR_Z']].values) * 1e3

    # Check if error columns exist in the EST data
    error_cols = ['EST_X_error', 'EST_Y_error', 'EST_Z_error']
    has_errors = all(col in final.columns for col in error_cols)

    if has_errors:
        logger.info("Error columns found in data, will process error propagation")
        # Convert errors from meters to millimeters for consistency with displacements
        for col in error_cols:
            final[col] = final[col] * 1e3
    else:
        logger.warning("Error columns not found in data, skipping error propagation")

    # Check for covariance columns
    cov_cols = ['EST_XY_COV', 'EST_XZ_COV', 'EST_YZ_COV']
    has_cov = all(col in final.columns for col in cov_cols)

    if has_cov:
        logger.info("Covariance columns found in data, will include in error propagation")
        # Convert covariances from meters to millimeters for consistency
        for col in cov_cols:
            final[col] = final[col] * 1e6  # Square of 1e3 because we're dealing with variances
    else:
        logger.warning("Covariance columns not found, will assume zero covariances in error propagation")

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

        # Get station lat/lon
        try:
            stalatlon = latlon.loc[sta]
            logger.info(f"Station {sta} coordinates: Lat={stalatlon['Latitude']}, Lon={stalatlon['Longitude']}")
        except KeyError:
            logger.error(f"No lat/lon data found for station {sta}")
            continue

        # First convert to topocentric coordinates, then add periodic signals
        df_reset = df.reset_index()

        try:
            df_with_topo = ecef_to_topo(
                df_reset,
                latitude=stalatlon['Latitude'],
                longitude=stalatlon['Longitude'],
                logger=logger
            )

            # Process errors if available
            if has_errors:
                logger.info(f"Processing error propagation for station {sta}")
                df_with_topo = ecef_to_topo_errors(
                    df_with_topo,
                    latitude=stalatlon['Latitude'],
                    longitude=stalatlon['Longitude'],
                    logger=logger
                )

        except Exception as e:
            logger.error(f"Error converting to topocentric coordinates for station {sta}: {e}")
            continue

        # Now add periodic signals in the topocentric domain if requested
        if add_periodic:
            try:
                logger.info(f"Adding periodic signals in topocentric domain for station {sta}")
                # Load the ENU periodic model files instead of XYZ
                df_processed = add_periodic_signals_enu(
                    df_with_topo,
                    'EXT/ITRF_PERIODIC_MODEL/ITRF2020-1cpy-ENU-CF.dat',  # Use ENU file instead of XYZ
                    'EXT/ITRF_PERIODIC_MODEL/ITRF2020-2cpy-ENU-CF.dat',  # Use ENU file instead of XYZ
                    station_code=sta,
                    logger=logger
                )
            except Exception as e:
                logger.error(f"Error adding periodic signals for station {sta}: {e}")
                df_processed = df_with_topo  # Use topo data without periodic if error occurs
        else:
            logger.info(f"Skipping periodic signal addition for station {sta} as requested")
            df_processed = df_with_topo

        # Save results
        try:
            # Define columns to save based on whether errors are available
            save_cols = ['dX', 'dY', 'dZ', 'dN', 'dE', 'dU']
            if has_errors:
                save_cols.extend(['EST_X_error', 'EST_Y_error', 'EST_Z_error', 'dN_error', 'dE_error', 'dU_error'])
                if has_cov:
                    save_cols.extend(['EST_XY_COV', 'EST_XZ_COV', 'EST_YZ_COV'])

            result = df_processed.set_index(['EPOCH', 'CODE', 'SOLN', 'S'])[save_cols]
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

def ecef_to_topo_errors(df, latitude, longitude, logger=None):
    """
    Transform ECEF coordinate errors to topocentric (North, East, Up) errors.

    This function implements error propagation for the coordinate transformation
    from ECEF to topocentric coordinates, using the variance-covariance information.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing EST_X_error, EST_Y_error, EST_Z_error columns
        and optionally the covariance components
    latitude : float
        Latitude of the station in degrees
    longitude : float
        Longitude of the station in degrees
    logger : logging.Logger, optional
        Logger object for logging messages

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional dN_error, dE_error, dU_error columns
    """
    # Use provided logger or create a simple one if not provided
    log = logger or logging.getLogger(__name__)

    log.info(f"Converting ECEF errors to topocentric error coordinates for lat={latitude}, lon={longitude}")

    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()

    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)

    # Precompute trigonometric values
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Rotation matrix elements
    # R = [ -sin_lat*cos_lon  -sin_lat*sin_lon  cos_lat ]
    #     [ -sin_lon           cos_lon          0       ]
    #     [  cos_lat*cos_lon   cos_lat*sin_lon  sin_lat ]

    # Check if covariance components are available
    has_cov = all(col in result_df.columns for col in ['EST_XY_COV', 'EST_XZ_COV', 'EST_YZ_COV'])

    # Process errors for each row
    dN_error = []
    dE_error = []
    dU_error = []

    for _, row in result_df.iterrows():
        # Get variance values (square of standard errors)
        var_x = row['EST_X_error'] ** 2
        var_y = row['EST_Y_error'] ** 2
        var_z = row['EST_Z_error'] ** 2

        # Get covariance values if available, otherwise assume zero
        if has_cov:
            cov_xy = row['EST_XY_COV']
            cov_xz = row['EST_XZ_COV']
            cov_yz = row['EST_YZ_COV']
        else:
            cov_xy = cov_xz = cov_yz = 0

        # Error propagation using the law of error propagation
        # var_NEU = R * var_XYZ * R^T where R is the rotation matrix

        # Calculate variances for N, E, U components
        r11 = -sin_lat * cos_lon
        r12 = -sin_lat * sin_lon
        r13 = cos_lat

        r21 = -sin_lon
        r22 = cos_lon
        r23 = 0

        r31 = cos_lat * cos_lon
        r32 = cos_lat * sin_lon
        r33 = sin_lat

        # North component variance
        var_n = (r11 ** 2 * var_x +
                 r12 ** 2 * var_y +
                 r13 ** 2 * var_z +
                 2 * r11 * r12 * cov_xy +
                 2 * r11 * r13 * cov_xz +
                 2 * r12 * r13 * cov_yz)

        # East component variance
        var_e = (r21 ** 2 * var_x +
                 r22 ** 2 * var_y +
                 r23 ** 2 * var_z +
                 2 * r21 * r22 * cov_xy +
                 2 * r21 * r23 * cov_xz +
                 2 * r22 * r23 * cov_yz)

        # Up component variance
        var_u = (r31 ** 2 * var_x +
                 r32 ** 2 * var_y +
                 r33 ** 2 * var_z +
                 2 * r31 * r32 * cov_xy +
                 2 * r31 * r33 * cov_xz +
                 2 * r32 * r33 * cov_yz)

        # Calculate standard errors (square root of variances)
        dN_error.append(np.sqrt(var_n))
        dE_error.append(np.sqrt(var_e))
        dU_error.append(np.sqrt(var_u))

    # Add new error columns to the DataFrame
    result_df['dN_error'] = dN_error  # North displacement error
    result_df['dE_error'] = dE_error  # East displacement error
    result_df['dU_error'] = dU_error  # Up displacement error

    log.info("ECEF to topocentric error conversion completed")
    return result_df

def add_periodic_signals_enu(df, cpy1_file, cpy2_file, station_code=None, reference_epoch='2015-01-01 00:00', logger=None):
    """
    Add periodic signals to dN, dE, and dU columns in a dataframe based on coefficients from
    ITRF2020 files for annual (1cpy) and semi-annual (2cpy) periodic signals in the
    topocentric (East, North, Up) coordinate system.

    The function processes each station and SOLN combination separately, applying
    the appropriate coefficients from the files.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'EPOCH' (datetime), 'CODE', 'SOLN', and dN, dE, dU columns
    cpy1_file : str
        Path to the ITRF2020-1cpy-ENU-CF.dat file (annual signal)
    cpy2_file : str
        Path to the ITRF2020-2cpy-ENU-CF.dat file (semi-annual signal)
    station_code : str, optional
        Four-character station code to filter by. If None, will use the 'CODE' column from the dataframe
    reference_epoch : str, optional
        Reference epoch for the periodic signals, default is '2015-01-01 00:00'
    logger : logging.Logger, optional
        Logger object for logging messages

    Returns:
    --------
    pandas.DataFrame
        DataFrame with updated dN, dE, dU columns that include the periodic signals
    """
    # Use provided logger or create a simple one if not provided
    log = logger or logging.getLogger(__name__)

    log.info(f"Adding periodic signals in ENU for station {station_code}")

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

    # Rename columns to match expected format
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

        # Components mapping - now using ENU components
        components = {'E': 'dE', 'N': 'dN', 'U': 'dU'}

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

    log.info(f"Completed adding periodic signals in ENU for station {station_code}")
    # Return the combined result with index reset
    return output_df.reset_index(drop=True)

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