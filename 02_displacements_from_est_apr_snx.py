import os
from glob import glob
from datetime import datetime
from snx_to_pkl import filter_files_by_date
import matplotlib
import pandas as pd
import numpy as np
import statsmodels.api as sm
from math import sin, cos, sqrt, pi
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from geodezyx.files_rw.geo_files_converter_lib import read_sinex_discontinuity_solo

import pandas as pd
import numpy as np
from datetime import datetime


def add_periodic_signals(df, cpy1_file, cpy2_file, station_code=None, reference_epoch='2015-01-01 00:00'):
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

    Returns:
    --------
    pandas.DataFrame
        DataFrame with updated dX, dY, dZ columns that include the periodic signals
    """
    # Make a copy of the input dataframe to avoid modifying the original
    result_df = df.copy()

    # Ensure EPOCH is datetime
    if not pd.api.types.is_datetime64_any_dtype(result_df['EPOCH']):
        result_df['EPOCH'] = pd.to_datetime(result_df['EPOCH'])

    # Convert reference epoch to datetime
    ref_epoch = pd.to_datetime(reference_epoch)

    # Define periods in days
    period_1cpy = 365.25  # Annual (1 cycle per year)
    period_2cpy = 182.625  # Semi-annual (2 cycles per year)

    # Read coefficient files
    coef_1cpy = pd.read_csv(cpy1_file, delim_whitespace=True,skiprows=3)
    coef_2cpy = pd.read_csv(cpy2_file, delim_whitespace=True,skiprows=3)

    coef_1cpy.columns = ['CODE', 'X', 'DOMES', 'SOLN', 'COMP', 'COSX', 'COSX_ERROR', 'SINX', 'SINX_ERROR']
    coef_2cpy.columns = ['CODE', 'X', 'DOMES', 'SOLN', 'COMP', 'COSX', 'COSX_ERROR', 'SINX', 'SINX_ERROR']

    # Ensure we have SOLN column in the dataframe
    if 'SOLN' not in result_df.columns:
        raise ValueError("Input dataframe must have a 'SOLN' column to match with coefficient files")

    # Create a copy to hold the result
    output_df = pd.DataFrame()

    # Process each SOLN group separately
    for soln, group_df in result_df.groupby('SOLN'):
        # Get a copy to work with
        temp_df = group_df.copy()

        # Filter coefficients for the specified station and solution number
        soln_coef_1cpy = coef_1cpy[(coef_1cpy['CODE'] == station_code) & (coef_1cpy['SOLN'] == soln)]
        soln_coef_2cpy = coef_2cpy[(coef_2cpy['CODE'] == station_code) & (coef_2cpy['SOLN'] == soln)]

        if soln_coef_1cpy.empty and soln_coef_2cpy.empty:
            print(f"Warning: No coefficients found for station code '{station_code}' and SOLN {soln}")
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

    # Return the combined result with index reset
    return output_df.reset_index(drop=True)


import numpy as np
import pandas as pd


def ecef_to_topo(df, latitude, longitude):
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

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with additional dN, dE, dU columns
    """
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

    return result_df

# solution = 'IGS1R03SNX'
solution = 'TUG0R03FIN'
sampling = '01D'

out_dir_code = rf'INPUT_CRD/{solution}_{sampling}/CODE/'
out_dir_time = rf'INPUT_CRD/{solution}_{sampling}/TIME/'
os.makedirs(out_dir_code,exist_ok=True)
os.makedirs(out_dir_time,exist_ok=True)

files_apr = glob(f'INPUT_CRD/{solution}/{solution}*{sampling}*APR.PKL')
files_est = glob(f'INPUT_CRD/{solution}/{solution}*{sampling}*EST.PKL')

# Define start and end dates
start_date = datetime(2000, 1, 1)
end_date = datetime(2024, 1, 1)

files_apr = filter_files_by_date(files_apr, start_date, end_date)
files_est = filter_files_by_date(files_est, start_date, end_date)

apr = pd.concat([pd.read_pickle(x) for x in files_apr],axis=0)
est = pd.concat([pd.read_pickle(x) for x in files_est],axis=0)

final = pd.concat([apr,est],axis=1)
final.loc[:,['dX','dY','dZ']] = (final[['EST_X','EST_Y','EST_Z']].values - final[['APR_X','APR_Y','APR_Z']].values)*1e3

latlon = pd.read_pickle(r'DATA/ALL_STATIONS_LATLON.pkl')

dfsta = final.groupby('CODE')
# #
results = []
for sta, df in dfsta:
    print(f'Processing {sta}')
    df_with_periodic = add_periodic_signals(df.reset_index(),
                                         'ITRF_PERIODIC_MODEL/ITRF2020-1cpy-XYZ-CF.dat',
                                         'ITRF_PERIODIC_MODEL/ITRF2020-2cpy-XYZ-CF.dat',
                                            station_code=sta)

    stalatlon = latlon.loc[sta]

    df_with_periodic_and_topo = ecef_to_topo(df_with_periodic, latitude=stalatlon['Latitude'],longitude = stalatlon['Longitude'])
    # validation = pd.read_pickle(r"C:\Users\rados\Documents\VM_SHARED\PYTHON\GFZLOAD2SNX\INPUT_CRD\IGS1R03SNX_01D\CODE\IGS1R03SNX_00NA_01D_DISP.PKL")
    result = df_with_periodic_and_topo.set_index(['EPOCH','CODE','SOLN','S'])[['dX','dY','dZ','dN','dE','dU']]

    result.to_pickle(os.path.join(out_dir_code,f'{solution}_{sta}_{sampling}_DISP.PKL'))
    results.append(result)

results_df = pd.concat(results,axis=0)

dftime = results_df.groupby('EPOCH')

for time, df in dftime:
    str_time = time.strftime('%Y%m%d')

    df.sort_index(level='CODE').to_pickle(os.path.join(out_dir_time,f'{solution}_{str_time}_{sampling}_DISP.PKL'))
# plot_displacements_by_solution(result2, components=['dX','dY','dZ'])

