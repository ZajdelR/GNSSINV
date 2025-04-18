import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from toolbox_gfzload2pkl import combine_selected_files
import os
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
import glob
import sys
from pathlib import Path
from geodezyx.conv.conv_time import dt2gpstime, gpstime2dt
import datetime
import concurrent.futures
import threading
import time
from gcc_analysis_tools import apply_bandpass_filter

# Add a lock for thread-safe file operations and printing
print_lock = threading.Lock()
file_lock = threading.Lock()

import matplotlib
matplotlib.use('Agg', force=True)  # Use non-interactive backend
plt.ioff()  # Turn off interactive mode

# Optionally, add this somewhere near imports to suppress any Tkinter warnings
import warnings
warnings.filterwarnings("ignore", ".*Tk*")


def calculate_period_days(cutoff_freq):
    """
    Safely calculate period in days from a cutoff frequency.

    Parameters:
    -----------
    cutoff_freq : float or None
        Cutoff frequency in Hz

    Returns:
    --------
    str or int
        Period in days, or "inf" if calculation not possible
    """
    if cutoff_freq is None:
        return "inf"

    try:
        # Convert frequency to period in days
        period_days = int(1 / cutoff_freq / 86400)
        return period_days
    except (ZeroDivisionError, ValueError, TypeError):
        # Handle invalid frequency values
        return "inf"

def find_stations(solution, sampling):
    """
    Find all stations available in the data directory.

    Parameters:
    -----------
    solution : str
        Solution name
    sampling : str
        Sampling rate

    Returns:
    --------
    List of station names
    """
    pattern = f'DATA/DISPLACEMENTS/{solution}_{sampling}/CODE/{solution}_*_{sampling}_DISP.PKL'
    files = glob.glob(pattern)

    stations = []
    for file in files:
        # Extract station name from filename
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 4:
            station = parts[1]  # Assuming filename format is solution_STA_sampling_DISP.PKL
            stations.append(station)

    return sorted(stations)

def process_station_wrapper(args):
    """
    Wrapper function to unpack arguments for process_station when using with ThreadPoolExecutor.

    Parameters:
    -----------
    args : tuple
        Tuple containing (sta, sampling, solution, reduce_components, compare_components,
                          filter_params, start_date, end_date)

    Returns:
    --------
    tuple
        (station_name, result) or (station_name, None) if processing failed
    """
    sta, sampling, solution, reduce_components, compare_components, filter_params, start_date, end_date = args
    try:
        result = process_station(
            sta, sampling, solution, reduce_components, compare_components,
            filter_params, start_date, end_date
        )
        return (sta, result)
    except Exception as e:
        with print_lock:
            print(f"Error processing station {sta}: {str(e)}")
        return (sta, None)


def process_station(sta, sampling, solution, reduce_components, compare_components,
                    filter_params, start_date=None, end_date=None):
    """
    Process a single station and create comparison plots.

    Parameters:
    -----------
    sta : str
        Station name
    sampling : str
        Sampling rate
    solution : str
        Solution name
    reduce_components : dict
        Dictionary indicating which components to remove from the original data
    compare_components : dict
        Dictionary indicating which components to include in the comparison sum
    filter_params : dict
        Dictionary containing filter parameters (lowcut, highcut, order)
    start_date : datetime.date, optional
        Start date for filtering data
    end_date : datetime.date, optional
        End date for filtering data

    Returns:
    --------
    dict or None
        Dictionary containing the comparison data, or None if processing failed
    """
    with print_lock:
        print(f"\nProcessing station: {sta}")

    # Extract filter parameters
    apply_filter = filter_params.get('apply_filter', False)
    lowcut = filter_params.get('lowcut', None)
    highcut = filter_params.get('highcut', None)
    filter_order = filter_params.get('order', 2)

    # Create output directory name and filter string based on filter parameters
    filter_str = ""
    if apply_filter:
        # Determine filter type based on provided cut frequencies
        if lowcut is not None and highcut is not None:
            # Band-pass filter
            subfolder = f"BP_{int(1 / highcut / 86400)}d_{int(1 / lowcut / 86400)}d"
            filter_str = f"_BP_{int(1 / highcut / 86400)}d_{int(1 / lowcut / 86400)}d"
            with print_lock:
                print(f"Applying band-pass filter ({1 / highcut / 86400:.1f}d - {1 / lowcut / 86400:.1f}d) to main data")
        elif lowcut is not None and highcut is None:
            # High-pass filter (passes frequencies above lowcut)
            subfolder = f"HP_{int(1 / lowcut / 86400)}d"
            filter_str = f"_HP_{int(1 / lowcut / 86400)}d"
            with print_lock:
                print(f"Applying high-pass filter (>{1 / lowcut / 86400:.1f}d) to main data")
        elif lowcut is None and highcut is not None:
            # Low-pass filter (passes frequencies below highcut)
            subfolder = f"LP_{int(1 / highcut / 86400)}d"
            filter_str = f"_LP_{int(1 / highcut / 86400)}d"
            with print_lock:
                print(f"Applying low-pass filter (<{1 / highcut / 86400:.1f}d) to main data")
        else:
            # No filtering
            subfolder = "NOFILTER"
            with print_lock:
                print("No filter parameters specified, skipping filtering")
    else:
        subfolder = "NOFILTER"

    # Create output directory if it doesn't exist
    output_dir = f'OUTPUT/SNX_LOAD_COMPARISONS/{solution}_{sampling}'
    with file_lock:
        os.makedirs(os.path.join(output_dir, 'PKL'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'TS_COMP'), exist_ok=True)

    # Load main displacement data
    df = load_station_data(sta, sampling, solution)
    if df is None:
        return None

    # Apply filter to main data if requested
    if apply_filter and (lowcut is not None or highcut is not None):
        # Calculate sampling rate based on sampling parameter
        try:
            sampling_days = int(sampling.strip('D'))
            sampling_rate = 1 / (sampling_days * 24 * 60 * 60)  # Convert to Hz
        except (ValueError, AttributeError):
            sampling_days = 1
            sampling_rate = 1 / (24 * 60 * 60)  # Default to daily sampling (Hz)

        with print_lock:
            print(f"Sampling rate: {sampling_rate} Hz (every {sampling_days} days)")

        filtered_df = pd.DataFrame(index=df.index)

        # Apply the filter to each column
        for column in ['dN', 'dE', 'dU']:
            # Check if column exists
            if column in df.columns:
                try:
                    filtered_df[column] = apply_bandpass_filter(
                        df[column].values,
                        lowcut,
                        highcut,
                        sampling_rate,
                        order=filter_order
                    )
                    with print_lock:
                        print(f"Successfully filtered {column}")
                except Exception as e:
                    with print_lock:
                        print(f"Error filtering {column}: {e}")
                    # Fallback to original values if filter fails
                    filtered_df[column] = df[column]
            else:
                with print_lock:
                    print(f"Column {column} not found in DataFrame")

        # Update the original dataframe with filtered values
        df.loc[:, ['dN', 'dE', 'dU']] = filtered_df.loc[:, ['dN', 'dE', 'dU']]

    # STEP 1: Handle the components to reduce from the original data
    df_red, reduce_components_name = reduce_components_from_data(
        sta, df, reduce_components, sampling, apply_filter, filter_params
    )
    if df_red is None:
        return None

    # STEP 2: Generate the comparison dataset by summing selected components
    comp_df, compare_components_name = create_comparison_data(
        sta, compare_components, sampling, apply_filter, filter_params
    )
    if comp_df is None:
        return None

    # Filter data by date range if specified
    if start_date is not None:
        df_red = df_red[df_red.index >= start_date]
        with print_lock:
            print(f"Filtered df_red from {start_date}, {len(df_red)} points remain")

    if end_date is not None:
        df_red = df_red[df_red.index <= end_date]
        with print_lock:
            print(f"Filtered df_red until {end_date}, {len(df_red)} points remain")

    # Do the same for the comparison dataset
    if start_date is not None and comp_df is not None:
        comp_df = comp_df[comp_df.index >= start_date]

    if end_date is not None and comp_df is not None:
        comp_df = comp_df[comp_df.index <= end_date]

    # Find common dates between df_red and comp_df
    common_dates = df_red.index.intersection(comp_df.index)
    with print_lock:
        print(
            f"Found {len(common_dates)} common dates between the df-{reduce_components_name} and SUM-{compare_components_name} datasets.")

    if len(common_dates) < 10:
        with print_lock:
            print(f"Warning: Too few common dates ({len(common_dates)}) for station {sta}. Skipping.")
        return None

    # Filter to common dates
    df_common = df_red.loc[common_dates]
    comp_common = comp_df.loc[common_dates]

    # Remove duplicated index values if any
    df_common = df_common[~df_common.index.duplicated(keep='first')]

    # Calculate statistics
    stats = calculate_statistics(df_common, comp_common, common_dates)
    differences_dU = stats['differences']

    # Calculate standard deviations for original data and individual components
    std_original_df = df.loc[common_dates]['dU'].std()
    std_comparison = comp_common['dU'].std()
    reduction_percent = (1 - stats['rms'] / std_original_df) * 100

    # Calculate standard deviations for each individual component
    component_stds = calculate_component_stds(sta, common_dates, apply_filter, filter_params)

    # Create plots
    fig = create_comparison_plots(sta, df_common, comp_common, differences_dU, stats,
                                 reduce_components_name, f"SUM-{compare_components_name}", sampling,
                                 std_original_df)

    # Save the comparison data
    comparison_data = {
        'df_reduced': df_common,
        'compare_component': comp_common,
        'differences': differences_dU,
        'reduce_components': reduce_components_name,
        'compare_components': compare_components_name,
        'stats': stats,
        'reduction_percent': reduction_percent,
        'filter_params': filter_params if apply_filter else None,
        'standard_deviations': {
            'original_df': std_original_df,
            'df_reduced': df_common['dU'].std(),
            'comparison_component': std_comparison,
            'individual_components': component_stds
        }
    }

    # Create filename based on what's being compared
    output_file = os.path.join(output_dir, 'PKL',
                              f'{solution}_{sta}_WO-{reduce_components_name}_VS_SUM-{compare_components_name}{filter_str}.PKL')

    with file_lock:
        pd.to_pickle(comparison_data, output_file)
        with print_lock:
            print(f"Comparison data saved to {output_file}")

    # Save the figure as PNG as well
    fig_output = os.path.join(output_dir, 'TS_COMP',
                             f'{solution}_{sta}_{sampling}_WO-{reduce_components_name}_VS_SUM-{compare_components_name}{filter_str}.png')

    with file_lock:
        plt.savefig(fig_output, dpi=200, bbox_inches='tight')
        with print_lock:
            print(f"Figure saved to {fig_output}")

    # Close the figure to free memory
    plt.close(fig)

    # Print basic statistics for reference
    with print_lock:
        print(
            f"\nStatistics of consistency between df-{reduce_components_name} and SUM-{compare_components_name} for dU:")
        print(f"Mean difference: {stats['mean']:.4f} mm")
        print(f"Median difference: {stats['median']:.4f} mm")
        print(f"Standard deviation: {stats['std']:.4f} mm")
        print(f"RMS difference: {stats['rms']:.4f} mm")
        print(f"Reduction: {reduction_percent:.1f}% of original std")
        print(f"Min/Max difference: {stats['min']:.4f}/{stats['max']:.4f} mm")
        print(f"Pearson correlation: {stats['correlation']:.4f} (p-value: {stats['p_value']:.2e})")
        print(f"Variance explained by SUM-{compare_components_name} (R²): {stats['variance_explained']:.2f}%")
        print(f"Number of common data points: {stats['num_points']}")

        print("\nStandard Deviations:")
        print(f"Original df: {std_original_df:.4f} mm")
        print(f"df-{reduce_components_name}: {df_common['dU'].std():.4f} mm")
        print(f"SUM-{compare_components_name}: {std_comparison:.4f} mm")
        print("\nIndividual Component Standard Deviations:")
        for component, std_val in component_stds.items():
            print(f"Component {component}: {std_val:.4f} mm")

    return comparison_data


def reduce_components_from_data(sta, df, reduce_components, sampling, apply_filter=False, filter_params=None):
    """
    Remove selected components from the original data.

    Parameters:
    -----------
    sta : str
        Station name
    df : pandas.DataFrame
        Original displacement data
    reduce_components : dict
        Dictionary indicating which components to remove
    sampling : str
        Sampling rate
    apply_filter : bool
        Whether to apply filter to the components
    filter_params : dict
        Dictionary containing filter parameters (lowcut, highcut, order)

    Returns:
    --------
    tuple
        (reduced_df, components_name) where reduced_df is the DataFrame with components removed
        and components_name is a string representation of the removed components
    """
    # Create list of components to remove
    component_labels = []
    files = []

    # Define all possible components
    components = ['A', 'O', 'S', 'H']

    # Always use the CODE folder for component data (not CODE_BP)
    # Loop through each component
    for component in components:
        # Only process if this component is enabled in reduce_components
        if reduce_components.get(component, False):
            file_path = f'EXT/ESMGFZLOADING/CODE/{sta}_{component}_cf.pkl'
            if os.path.exists(file_path):
                files.append(file_path)
                component_labels.append(component)
            else:
                with print_lock:
                    print(f"Warning: Component file not found: {file_path}")

    # Convert index to date for consistency
    df.index = df.index.date

    # Load and combine selected components to remove
    if files:
        with file_lock:
            files_df = {os.path.basename(x): pd.read_pickle(x) for x in files}
        sum_df, name = combine_selected_files(files_df)
        sum_df = sum_df.rename({'R': 'dU', 'NS': 'dN', 'EW': 'dE'}, axis=1)

        # Convert index to date for consistency
        sum_df.index = sum_df.index.date

        # Apply filter to component data if requested
        if apply_filter and filter_params:
            # Extract filter parameters
            lowcut = filter_params.get('lowcut', None)
            highcut = filter_params.get('highcut', None)
            filter_order = filter_params.get('order', 2)

            # Only apply filter if at least one cutoff frequency is specified
            if lowcut is not None or highcut is not None:
                # Determine filter type for display purposes
                if lowcut is not None and highcut is not None:
                    filter_type = "band-pass"
                    filter_desc = f"({1 / highcut / 86400:.1f}d - {1 / lowcut / 86400:.1f}d)"
                elif lowcut is not None and highcut is None:
                    filter_type = "high-pass"
                    filter_desc = f"(>{1 / lowcut / 86400:.1f}d)"
                elif lowcut is None and highcut is not None:
                    filter_type = "low-pass"
                    filter_desc = f"(<{1 / highcut / 86400:.1f}d)"

                with print_lock:
                    print(f"Applying {filter_type} filter {filter_desc} to reduction components: {component_labels}")

                # Calculate sampling rate
                try:
                    sampling_days = int(sampling.strip('D'))
                    sampling_rate = 1 / (sampling_days * 24 * 60 * 60)  # Convert to Hz
                except (ValueError, AttributeError):
                    sampling_days = 1
                    sampling_rate = 1 / (24 * 60 * 60)  # Default to daily sampling (Hz)

                # Apply the filter to each column
                for column in ['dN', 'dE', 'dU']:
                    if column in sum_df.columns:
                        try:
                            sum_df[column] = apply_bandpass_filter(
                                sum_df[column].values,
                                lowcut,
                                highcut,
                                sampling_rate,
                                order=filter_order
                            )
                            with print_lock:
                                print(f"Successfully filtered {column} for reduction components")
                        except Exception as e:
                            with print_lock:
                                print(f"Error filtering {column} for reduction components: {e}")
                    else:
                        with print_lock:
                            print(f"Column {column} not found in reduction components DataFrame")

        if sampling == '07D':
            sum_df[['gpsweek', 'doy']] = pd.DataFrame(
                sum_df.index.map(
                    lambda x: dt2gpstime(datetime.datetime.combine(x, datetime.time()))
                ).tolist(),
                index=sum_df.index
            )
            sum_df = aggregate_gps_data(sum_df)

            df[['gpsweek', 'doy']] = pd.DataFrame(
                df.index.map(
                    lambda x: dt2gpstime(datetime.datetime.combine(x, datetime.time()))
                ).tolist(),
                index=df.index
            )
            df = aggregate_gps_data(df)

        # Remove summed components from main data
        df_red = (df - sum_df).dropna()
        components_name = ''.join(component_labels)
    else:
        # If no components selected for removal, use original data
        df_red = df
        components_name = "None"

    return df_red, components_name

def create_comparison_data(sta, compare_components, sampling, apply_filter=False, filter_params=None):
    """
    Create a comparison dataset by summing selected components.

    Parameters:
    -----------
    sta : str
        Station name
    compare_components : dict
        Dictionary indicating which components to include in the sum
    sampling : str
        Sampling rate
    apply_filter : bool
        Whether to apply filter to the components
    filter_params : dict
        Dictionary containing filter parameters (lowcut, highcut, order)

    Returns:
    --------
    tuple
        (comp_df, components_name) where comp_df is the summed DataFrame
        and components_name is a string representation of the included components
    """
    # Create list of components to include in sum
    component_labels = []
    files = []

    # Define all possible components
    components = ['A', 'O', 'S', 'H', 'M', 'L']
    # Always use the CODE folder for component data (not CODE_BP)
    # Loop through each component
    for component in components:
        # Only process if this component is enabled in compare_components
        if compare_components.get(component, False):
            file_path = f'EXT/ESMGFZLOADING/CODE/{sta}_{component}_cf.pkl'
            if os.path.exists(file_path):
                files.append(file_path)
                component_labels.append(component)
            else:
                with print_lock:
                    print(f"Warning: Component file not found: {file_path}")

    # Load and combine selected components
    if files:
        with file_lock:
            files_df = {os.path.basename(x): pd.read_pickle(x) for x in files}
        sum_df, name = combine_selected_files(files_df)
        sum_df = sum_df.rename({'R': 'dU', 'NS': 'dN', 'EW': 'dE'}, axis=1)

        # Convert index to date for consistency
        sum_df.index = sum_df.index.date

        # Apply filter to component data if requested
        if apply_filter and filter_params:
            # Extract filter parameters
            lowcut = filter_params.get('lowcut', None)
            highcut = filter_params.get('highcut', None)
            filter_order = filter_params.get('order', 2)

            # Only apply filter if at least one cutoff frequency is specified
            if lowcut is not None or highcut is not None:
                # Determine filter type for display purposes
                if lowcut is not None and highcut is not None:
                    filter_type = "band-pass"
                    filter_desc = f"({1 / highcut / 86400:.1f}d - {1 / lowcut / 86400:.1f}d)"
                elif lowcut is not None and highcut is None:
                    filter_type = "high-pass"
                    filter_desc = f"(>{1 / lowcut / 86400:.1f}d)"
                elif lowcut is None and highcut is not None:
                    filter_type = "low-pass"
                    filter_desc = f"(<{1 / highcut / 86400:.1f}d)"

                with print_lock:
                    print(f"Applying {filter_type} filter {filter_desc} to comparison components: {component_labels}")
                    # Calculate sampling rate
                    try:
                        sampling_days = int(sampling.strip('D'))
                        sampling_rate = 1 / (sampling_days * 24 * 60 * 60)  # Convert to Hz
                    except (ValueError, AttributeError):
                        sampling_days = 1
                        sampling_rate = 1 / (24 * 60 * 60)  # Default to daily sampling (Hz)

                    # Apply the filter to each column
                    for column in ['dN', 'dE', 'dU']:
                        if column in sum_df.columns:
                            try:
                                sum_df[column] = apply_bandpass_filter(
                                    sum_df[column].values,
                                    lowcut,
                                    highcut,
                                    sampling_rate,
                                    order=filter_order
                                )
                                with print_lock:
                                    print(f"Successfully filtered {column} for comparison components")
                            except Exception as e:
                                with print_lock:
                                    print(f"Error filtering {column} for comparison components: {e}")
                        else:
                            with print_lock:
                                print(f"Column {column} not found in comparison components DataFrame")

                if sampling == '07D':
                    sum_df[['gpsweek', 'doy']] = pd.DataFrame(
                        sum_df.index.map(
                            lambda x: dt2gpstime(datetime.datetime.combine(x, datetime.time()))
                        ).tolist(),
                        index=sum_df.index
                    )
                    sum_df = aggregate_gps_data(sum_df)

                components_name = ''.join(component_labels)
                return sum_df, components_name
            else:
                # If no components selected, return None
                with print_lock:
                    print(f"Warning: No components selected for comparison for station {sta}")
                return None, "None"


def create_comparison_data(sta, compare_components, sampling, apply_filter=False, filter_params=None):
    """
    Create a comparison dataset by summing selected components.

    Parameters:
    -----------
    sta : str
        Station name
    compare_components : dict
        Dictionary indicating which components to include in the sum
    sampling : str
        Sampling rate
    apply_filter : bool
        Whether to apply filter to the components
    filter_params : dict
        Dictionary containing filter parameters (lowcut, highcut, order)

    Returns:
    --------
    tuple
        (comp_df, components_name) where comp_df is the summed DataFrame
        and components_name is a string representation of the included components
    """
    # Create list of components to include in sum
    component_labels = []
    files = []

    # Define all possible components
    components = ['A', 'O', 'S', 'H', 'M', 'L']

    # Always use the CODE folder for component data (not CODE_BP)
    # Loop through each component
    for component in components:
        # Only process if this component is enabled in compare_components
        if compare_components.get(component, False):
            file_path = f'EXT/ESMGFZLOADING/CODE/{sta}_{component}_cf.pkl'
            if os.path.exists(file_path):
                files.append(file_path)
                component_labels.append(component)
            else:
                with print_lock:
                    print(f"Warning: Component file not found: {file_path}")

    # Load and combine selected components
    if files:
        with file_lock:
            files_df = {os.path.basename(x): pd.read_pickle(x) for x in files}
        sum_df, name = combine_selected_files(files_df)
        sum_df = sum_df.rename({'R': 'dU', 'NS': 'dN', 'EW': 'dE'}, axis=1)

        # Convert index to date for consistency
        sum_df.index = sum_df.index.date

        # Apply filter to component data if requested
        if apply_filter and filter_params:
            # Extract filter parameters
            lowcut = filter_params.get('lowcut', None)
            highcut = filter_params.get('highcut', None)
            filter_order = filter_params.get('order', 2)

            # Only apply filter if at least one cutoff frequency is specified
            if lowcut is not None or highcut is not None:
                # Determine filter type for display purposes
                if lowcut is not None and highcut is not None:
                    filter_type = "band-pass"
                    filter_desc = f"({1 / highcut / 86400:.1f}d - {1 / lowcut / 86400:.1f}d)"
                elif lowcut is not None and highcut is None:
                    filter_type = "high-pass"
                    filter_desc = f"(>{1 / lowcut / 86400:.1f}d)"
                elif lowcut is None and highcut is not None:
                    filter_type = "low-pass"
                    filter_desc = f"(<{1 / highcut / 86400:.1f}d)"

                with print_lock:
                    print(f"Applying {filter_type} filter {filter_desc} to comparison components: {component_labels}")

                # Calculate sampling rate
                try:
                    sampling_days = int(sampling.strip('D'))
                    sampling_rate = 1 / (sampling_days * 24 * 60 * 60)  # Convert to Hz
                except (ValueError, AttributeError):
                    sampling_days = 1
                    sampling_rate = 1 / (24 * 60 * 60)  # Default to daily sampling (Hz)

                # Apply the filter to each column
                for column in ['dN', 'dE', 'dU']:
                    if column in sum_df.columns:
                        try:
                            sum_df[column] = apply_bandpass_filter(
                                sum_df[column].values,
                                lowcut,
                                highcut,
                                sampling_rate,
                                order=filter_order
                            )
                            with print_lock:
                                print(f"Successfully filtered {column} for comparison components")
                        except Exception as e:
                            with print_lock:
                                print(f"Error filtering {column} for comparison components: {e}")
                    else:
                        with print_lock:
                            print(f"Column {column} not found in comparison components DataFrame")

        if sampling == '07D':
            sum_df[['gpsweek', 'doy']] = pd.DataFrame(
                sum_df.index.map(
                    lambda x: dt2gpstime(datetime.datetime.combine(x, datetime.time()))
                ).tolist(),
                index=sum_df.index
            )
            sum_df = aggregate_gps_data(sum_df)

        components_name = ''.join(component_labels)
        return sum_df, components_name
    else:
        # If no components selected, return None
        with print_lock:
            print(f"Warning: No components selected for comparison for station {sta}")
        return None, "None"

def aggregate_gps_data(df):
    """
    Aggregate GPS displacement data by GPS week and create a datetime index.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing columns: 'dU', 'dE', 'dN', 'gpsweek', 'doy'

    Returns:
    -------
    pandas.DataFrame
        Aggregated DataFrame with mean displacement values per GPS week
        and datetime index based on GPS week and doy = 3
    """
    # Group by gpsweek and calculate mean of displacement values
    agg_df = df.groupby('gpsweek')[['dU', 'dE', 'dN']].mean().reset_index()

    agg_df['date'] = agg_df['gpsweek'].apply(lambda x: gpstime2dt(x, 3)).dt.date
    agg_df.drop('gpsweek', axis=1, inplace=True)
    # Set the datetime column as index
    agg_df = agg_df.set_index('date')

    return agg_df


def load_station_data(sta, sampling, solution):
    """
    Load the main displacement data for a station.

    Parameters:
    -----------
    sta : str
        Station name
    sampling : str
        Sampling rate (e.g., '01D', '07D')
    solution : str
        Solution name

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing the displacement data, or None if file not found
    """
    try:
        file_path = f'DATA/DISPLACEMENTS/{solution}_{sampling}/CODE/{solution}_{sta}_{sampling}_DISP.PKL'
        with file_lock:
            df = pd.read_pickle(file_path)
        df = df.reset_index(level='EPOCH')[['EPOCH', 'dU', 'dN', 'dE']].set_index('EPOCH')
        return df
    except FileNotFoundError:
        with print_lock:
            print(f"Warning: Station data file not found for {sta}")
        return None
    except Exception as e:
        with print_lock:
            print(f"Error loading station data for {sta}: {str(e)}")
        return None


def load_component_data(sta, component, apply_filter=False, filter_params=None):
    """
    Load a specific loading component data for a station.

    Parameters:
    -----------
    sta : str
        Station name
    component : str
        Component name ('A', 'H', 'O', 'S')
    apply_filter : bool
        Whether to apply filter to the component
    filter_params : dict
        Dictionary containing filter parameters (lowcut, highcut, order)

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing the component data, or None if file not found
    """
    try:
        # Always load from CODE folder, not CODE_BP
        file_path = f'EXT/ESMGFZLOADING/CODE/{sta}_{component}_cf.pkl'
        with file_lock:
            df = pd.read_pickle(file_path)
        df = df.rename({'R': 'dU', 'NS': 'dN', 'EW': 'dE'}, axis=1)

        # Apply filter if requested
        if apply_filter and filter_params:
            # Extract filter parameters
            lowcut = filter_params.get('lowcut', None)
            highcut = filter_params.get('highcut', None)
            filter_order = filter_params.get('order', 2)
            sampling_rate = 1 / (24 * 60 * 60)  # Default to daily (Hz)

            # Only apply filter if at least one cutoff frequency is specified
            if lowcut is not None or highcut is not None:
                for column in ['dN', 'dE', 'dU']:
                    if column in df.columns:
                        try:
                            df[column] = apply_bandpass_filter(
                                df[column].values,
                                lowcut,
                                highcut,
                                sampling_rate,
                                order=filter_order
                            )
                        except Exception as e:
                            with print_lock:
                                print(f"Error filtering {column} for component {component}: {e}")

        return df
    except FileNotFoundError:
        with print_lock:
            print(f"Warning: Component {component} file not found for station {sta}")
        return None
    except Exception as e:
        with print_lock:
            print(f"Error loading component {component} for station {sta}: {str(e)}")
        return None

def calculate_statistics(df1, df2, common_dates):
    """
    Calculate statistics between two time series.

    Parameters:
    -----------
    df1 : pandas.DataFrame
        First dataframe
    df2 : pandas.DataFrame
        Second dataframe
    common_dates : list
        List of common dates between the two dataframes

    Returns:
    --------
    dict
        Dictionary containing the calculated statistics
    """
    import numpy as np
    from scipy import stats

    # Extract the dU values for common dates
    series1 = df1.loc[common_dates]['dU'] - np.mean(df1.loc[common_dates]['dU'])
    series2 = df2.loc[common_dates]['dU'] - np.mean(df2.loc[common_dates]['dU'])

    # Calculate differences
    differences = series1 - series2

    # Calculate statistics
    mean_diff = differences.mean()
    std_diff = differences.std()
    median_diff = differences.median()
    max_diff = differences.max()
    min_diff = differences.min()
    rms_diff = np.sqrt(np.mean(differences ** 2))

    # Calculate correlation
    corr, p_value = stats.pearsonr(series1, series2)
    r_squared = corr ** 2
    variance_explained = r_squared * 100

    # Calculate KGE (Kling-Gupta Efficiency) components
    # KGE has three components: correlation (r), bias ratio (beta), and variability ratio (alpha)
    mean_obs = series1.mean()
    mean_sim = series2.mean()
    std_obs = series1.std()
    std_sim = series2.std()

    # Calculate the components
    r = corr  # Correlation component (already calculated)
    beta = 1  # Bias ratio
    alpha = std_sim / std_obs  # Variability ratio

    # Calculate KGE (2009 version)
    kge_2009 = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (alpha - 1) ** 2)

    # Calculate modified KGE (2012 version)
    kge_2012 = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (alpha - 1) ** 2)

    return {
        'differences': differences,
        'mean': mean_diff,
        'median': median_diff,
        'std': std_diff,
        'rms': rms_diff,
        'min': min_diff,
        'max': max_diff,
        'correlation': corr,
        'p_value': p_value,
        'r_squared': r_squared,
        'variance_explained': variance_explained,
        'kge': kge_2009,  # Original KGE (Gupta et al., 2009)
        'kge_modified': kge_2012,  # Modified KGE (Kling et al., 2012)
        'kge_components': {
            'r': r,  # Correlation component
            'beta': beta,  # Bias ratio
            'alpha': alpha  # Variability ratio
        },
        'num_points': len(common_dates)
    }

def calculate_component_stds(sta, common_dates, apply_filter=False, filter_params=None):
    """
    Calculate standard deviations for each individual component.

    Parameters:
    -----------
    sta : str
        Station name
    common_dates : list
        List of common dates to use for calculation
    apply_filter : bool
        Whether to apply bandpass filter to the components
    filter_params : dict
        Dictionary containing bandpass filter parameters (lowcut, highcut, order)

    Returns:
    --------
    dict
        Dictionary containing the standard deviations for each component
    """
    component_stds = {}

    for component in ['A', 'O', 'S', 'H', 'M', 'L']:
        try:
            comp_df = load_component_data(sta, component, apply_filter, filter_params)
            if comp_df is None:
                component_stds[component] = np.nan
                continue

            comp_df.index = comp_df.index.date

            # Check if the component has data for common dates
            if set(common_dates).issubset(set(comp_df.index)):
                component_stds[component] = comp_df.loc[common_dates]['dU'].std()
            else:
                # Find intersection of dates if not all common dates are available
                comp_dates = comp_df.index.intersection(common_dates)
                if len(comp_dates) > 0:
                    component_stds[component] = comp_df.loc[comp_dates]['dU'].std()
                    with print_lock:
                        print(
                            f"Warning: Component {component} only has {len(comp_dates)} of {len(common_dates)} common dates")
                else:
                    component_stds[component] = np.nan
                    with print_lock:
                        print(f"Warning: Component {component} has no data for the common dates")
        except Exception as e:
            component_stds[component] = np.nan
            with print_lock:
                print(f"Could not calculate standard deviation for component {component}: {str(e)}")

    return component_stds


def create_comparison_plots(sta, df_common, comp_common, differences, stats, reduce_components_name,
                            compare_components_name, sampling, std_original_df):
    """
    Create comparison plots for a specific station with improved Lomb-Scargle implementation.

    Parameters:
    -----------
    sta : str
        Station name
    df_common : pandas.DataFrame
        DataFrame with reduced components
    comp_common : pandas.DataFrame
        DataFrame with components to compare
    differences : pandas.Series
        Series with differences between df_common and comp_common
    stats : dict
        Dictionary with statistics
    reduce_components_name : str
        String representation of components reduced from original data
    compare_components_name : str
        String representation of components being compared
    sampling : str
        Sampling rate
    std_original_df : float
        Standard deviation of the original data

    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the comparison plots
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.timeseries import LombScargle
    from scipy.optimize import curve_fit

    def annual_signal_model(t, amplitude, phase):
        """Simple annual signal model with fixed period (365.25 days)"""
        omega = 2 * np.pi / 365.25  # Angular frequency for annual cycle
        return amplitude * np.sin(omega * t + phase)

    def fit_annual_signal(times, values, annual_freq=1.0 / 365.25):
        """Fit a sine function with annual period to extract amplitude"""
        # Initial guess for parameters [amplitude, phase]
        p0 = [np.std(values), 0]

        # Angular frequency for annual cycle
        omega = 2 * np.pi * annual_freq

        try:
            # Function to fit
            def fit_func(t, amp, phase):
                return amp * np.sin(omega * t + phase)

            # Perform the fit
            popt, _ = curve_fit(fit_func, times, values, p0=p0)
            amplitude = np.abs(popt[0])  # Take absolute value of amplitude
            return amplitude
        except:
            # If fitting fails, return NaN
            return np.nan

    common_dates = df_common.index

    # Create a single figure with a grid of subplots
    fig = plt.figure(figsize=(10, 12))

    # Define grid with 3 rows, 3 columns (with different column widths)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1.5, 1], width_ratios=[2, 1, 0])

    # Time series subplot (upper left, 2/3 width)
    ax_timeseries = fig.add_subplot(gs[0, 0])
    ax_timeseries.plot(df_common.index, df_common['dU'], '-', linewidth=1,
                       label=f'df-{reduce_components_name} dU', color='blue', alpha=1, markersize=0)
    ax_timeseries.plot(comp_common.index, comp_common['dU'], '-', linewidth=1,
                       label=f'{compare_components_name} dU', color='red', alpha=0.5, markersize=0)

    # Find min and max values for setting symmetric ylim rounded to nearest multiple of 5
    all_values = np.concatenate([df_common['dU'].dropna().values, comp_common['dU'].dropna().values])
    max_abs_val = np.max(np.abs(all_values))
    # Round up to the nearest multiple of 5
    ylim = np.ceil(max_abs_val / 5) * 5

    ax_timeseries.set_ylim(-ylim, ylim)
    ax_timeseries.set_ylabel('dU Value (mm)')
    ax_timeseries.set_title(f'Time Series Comparison of dU for {sta}')
    ax_timeseries.grid(True, linestyle='--', alpha=0.6)
    ax_timeseries.legend(loc='best', fontsize=9)

    # Lomb-Scargle periodogram subplot (upper right, 1/3 width)
    ax_lomb = fig.add_subplot(gs[0, 1])

    # Parse sampling interval from the sampling variable (e.g., '01D' or '07D')
    # Extract the number part and convert to integer
    try:
        # Assume sampling is a string like '01D' or '07D'
        sampling_days = int(sampling.strip('D'))
        with print_lock:
            print(f"Detected sampling interval: {sampling_days} days")
    except (ValueError, AttributeError):
        # Default to 1 day if there's an issue parsing
        sampling_days = 1
        with print_lock:
            print(f"Warning: Couldn't parse sampling interval, defaulting to {sampling_days} day")

    # Calculate Nyquist period (minimum resolvable period = 2 * sampling interval)
    nyquist_period = 2 * sampling_days

    # Convert datetime to numerical values (days since first observation)
    dates_num1 = np.array([(d - common_dates[0]).total_seconds() / (24 * 3600) for d in common_dates])

    # Define frequency range that adapts to the sampling interval
    min_freq = 1.0 / 1000.0  # Lowest frequency (500-day period)
    max_freq = 1.0 / nyquist_period  # Highest frequency (adjusted based on sampling)

    # Determine appropriate number of frequency points based on sampling
    # More points for finer sampling, fewer for coarser sampling
    num_points = int(1000 / sampling_days) if sampling_days > 0 else 1000
    num_points = max(200, min(num_points, 1000))  # Keep between 200-1000 points

    # Create main frequency array (logarithmic spacing)
    freq = np.logspace(np.log10(min_freq), np.log10(max_freq), num_points)

    # Add extra points around annual and semi-annual periods
    # More detailed resolution for finer sampling
    # annual_points = int(200 / sampling_days) if sampling_days > 0 else 200
    # annual_points = max(50, min(annual_points, 200))  # Keep between 50-200 points
    # semiannual_points = int(annual_points / 2)
    #
    # freq_annual = np.linspace(1.0 / 385.0, 1.0 / 345.0, annual_points)  # Around annual period
    # freq_semiannual = np.linspace(1.0 / 192.0, 1.0 / 172.0, semiannual_points)  # Around semi-annual
    #
    # # Combine frequencies and sort
    # freq = np.unique(np.concatenate([freq, freq_annual, freq_semiannual]))

    # Extract data values and handle NaN values
    dU1 = df_common['dU'].values
    dU2 = comp_common['dU'].values

    # Create masks for non-NaN values
    mask1 = ~np.isnan(dU1)
    mask2 = ~np.isnan(dU2)

    valid_dates1 = dates_num1[mask1]
    valid_dU1 = dU1[mask1]
    valid_dates2 = dates_num1[mask2]
    valid_dU2 = dU2[mask2]

    # Check if we have enough data points
    min_points_required = 10  # Adjust based on your needs
    if len(valid_dates1) < min_points_required or len(valid_dates2) < min_points_required:
        with print_lock:
            print(
                f"Warning: Not enough valid data points. Series 1: {len(valid_dates1)}, Series 2: {len(valid_dates2)}")
        # Create empty plot with warning text
        ax_lomb.text(0.5, 0.5, "Insufficient data for Lomb-Scargle analysis",
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax_lomb.transAxes)
        # Set up basic plot elements and return
        ax_lomb.set_xlabel('Period (days)', fontsize=10, labelpad=8)
        ax_lomb.set_ylabel('Amplitude (mm)', fontsize=10)
        ax_lomb.set_title('Lomb-Scargle Periodogram')
        ax_lomb.grid(True, linestyle='--', alpha=0.6)
    else:
        # Create Lomb-Scargle objects
        ls1 = LombScargle(valid_dates1, valid_dU1)
        ls2 = LombScargle(valid_dates2, valid_dU2)

        # Calculate annual frequency
        annual_freq = 1.0 / 365.25

        # Fit single frequency models to get amplitude of annual signal
        annual_amp1 = fit_annual_signal(valid_dates1, valid_dU1, annual_freq)
        annual_amp2 = fit_annual_signal(valid_dates2, valid_dU2, annual_freq)

        # Print for debugging
        with print_lock:
            print(f"Fitted annual amplitudes: {annual_amp1:.2f} mm, {annual_amp2:.2f} mm")

        # Add annual amplitude information in the top left
        amp_text = (
            f"Annual amplitude:\n"
            f"df-{reduce_components_name}: {annual_amp1:.2f} mm\n"
            f"{compare_components_name}: {annual_amp2:.2f} mm"
        )
        ax_timeseries.text(0.02, 0.95, amp_text,
                           transform=ax_timeseries.transAxes,
                           verticalalignment='top',
                           horizontalalignment='left',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                           fontsize=9)

        std_text = (
            f"Std of series:\n"
            f"df-{reduce_components_name}: {valid_dU1.std():.2f} mm\n"
            f"{compare_components_name}: {valid_dU2.std():.2f} mm"
        )

        ax_timeseries.text(0.02, 0.20, std_text,
                           transform=ax_timeseries.transAxes,
                           verticalalignment='top',
                           horizontalalignment='left',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                           fontsize=9)

        # Add normalization option for better results with sparse data
        normalization = 'standard'  # Try 'standard', 'model', or 'log' if having issues

        # Compute power spectra at the specified frequencies
        power1 = ls1.power(freq, normalization=normalization)
        power2 = ls2.power(freq, normalization=normalization)

        # Convert to amplitudes manually
        # For normalized power spectrum, amplitude = sqrt(2*power) * std(signal)
        amplitude1 = np.sqrt(2 * power1) * np.std(valid_dU1)
        amplitude2 = np.sqrt(2 * power2) * np.std(valid_dU2)

        # Convert frequency to period (days)
        period = 1.0 / freq

        # Find amplitudes at key periods
        annual_idx = np.argmin(np.abs(period - 365.25))
        semiannual_idx = np.argmin(np.abs(period - 182.625))

        # Print key period amplitudes for debugging
        with print_lock:
            print(f"Annual amplitude: {amplitude1[annual_idx]:.2f} mm, {amplitude2[annual_idx]:.2f} mm")
            print(f"Semi-annual amplitude: {amplitude1[semiannual_idx]:.2f} mm, {amplitude2[semiannual_idx]:.2f} mm")

        # Plot the periodograms
        ax_lomb.plot(period, amplitude1, '-', color='blue', alpha=1, linewidth=1, label=f'df-{reduce_components_name}')
        ax_lomb.plot(period, amplitude2, '-', color='red', alpha=0.5, linewidth=1, label=f'{compare_components_name}')
        ax_lomb.set_xlabel('Period (days)', fontsize=10, labelpad=8)
        ax_lomb.set_ylabel('Amplitude (mm)', fontsize=10)
        ax_lomb.set_title('Lomb-Scargle Periodogram')
        ax_lomb.set_ylim(0, np.ceil(max([amplitude1[annual_idx], amplitude2[annual_idx]])))
        ax_lomb.grid(True, linestyle='--', alpha=0.6)
        ax_lomb.legend(loc='upper left', fontsize=9)
        ax_lomb.set_xscale('log')
        ax_lomb.set_xlim(nyquist_period, 500)  # Dynamically adjust x-axis limits based on sampling

    # Ensure x-axis labels are visible and appropriately sized
    ax_lomb.tick_params(axis='x', labelsize=9, pad=5)
    ax_lomb.tick_params(axis='y', labelsize=9)

    # Add more x-tick labels to better show the log scale
    ax_lomb.set_xticks([2, 5, 10, 20, 50, 100, 500,1000])
    ax_lomb.set_xticklabels(['2', '5', '10', '20', '50', '100', '500', '1000'])

    # Highlight important periods with vertical lines
    for period_val in [365.25, 182.625]:  # Annual and semi-annual
        ax_lomb.axvline(x=period_val, color='k', linestyle='--', alpha=0.5)

    # Highlight some common periods
    common_periods = [365.25, 182.625]  # Annual, semi-annual
    for period_val in common_periods:
        ax_lomb.axvline(x=period_val, color='gray', linestyle='--', alpha=0.5)

    # Differences subplot (middle row, left 2/3)
    ax_diff = fig.add_subplot(gs[1, 0], sharex=ax_timeseries)
    ax_diff.plot(common_dates, differences, 'o-', linewidth=0, color='purple', alpha=0.8, markersize=3)
    ax_diff.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_diff.fill_between(common_dates, differences, 0, where=differences > 0, color='green', alpha=0.3)
    ax_diff.fill_between(common_dates, differences, 0, where=differences < 0, color='red', alpha=0.3)

    # Set symmetric ylim for differences rounded to nearest multiple of 5
    diff_max = np.max(np.abs(differences.dropna().values))
    diff_ylim = np.ceil(diff_max / 5) * 5
    ax_diff.set_ylim(-diff_ylim, diff_ylim)

    ax_diff.set_ylabel(f'Difference (df-{reduce_components_name} - {compare_components_name})')
    ax_diff.set_title(f'Differences Between df-{reduce_components_name} and {compare_components_name} for dU')
    ax_diff.grid(True, linestyle='--', alpha=0.6)
    # ax_diff.set_xlim(datetime.datetime(2004,1,1),datetime.datetime(2008,1,1))

    # Update rms_diff_text to include reduction percentage
    reduction_percent = (1 - stats['rms'] / std_original_df) * 100
    rms_diff_text = (
        f"RMS of differences: {stats['rms']:.2f} mm\n"
        f"Reduction: {reduction_percent:.1f}% of orig. std"
    )

    # Lomb-Scargle periodogram for differences (middle row, right 1/3)
    ax_diff_lomb = fig.add_subplot(gs[1, 1])

    # Calculate Lomb-Scargle periodogram for differences
    diff_vals = differences.values
    mask_diff = ~np.isnan(diff_vals)
    valid_diff_dates = dates_num1[mask_diff]
    valid_diff_vals = diff_vals[mask_diff]

    # Create Lomb-Scargle object for differences
    ls_diff = LombScargle(valid_diff_dates, valid_diff_vals)
    power_diff = ls_diff.power(freq)
    amplitude_diff = np.sqrt(2 * power_diff) * np.std(valid_diff_vals)

    # Find key period amplitudes for differences
    diff_annual_amp = amplitude_diff[annual_idx]
    diff_semiannual_amp = amplitude_diff[semiannual_idx]

    # Find peak amplitude and its period in differences
    max_diff_idx = np.argmax(amplitude_diff)
    max_diff_period = period[max_diff_idx]
    max_diff_amp = amplitude_diff[max_diff_idx]

    with print_lock:
        print(f"Difference signal - Annual: {diff_annual_amp:.2f} mm, Semi-annual: {diff_semiannual_amp:.2f} mm")
        print(f"Difference signal - Maximum amplitude: {max_diff_amp:.2f} mm at period {max_diff_period:.2f} days")

    # Plot the periodogram for differences
    ax_diff_lomb.plot(period, amplitude_diff, '-', color='purple', alpha=0.7, label='Differences')
    ax_diff_lomb.set_xlabel('Period (days)', fontsize=10, labelpad=8)
    ax_diff_lomb.set_ylabel('Amplitude (mm)', fontsize=10)
    ax_diff_lomb.set_title('Lomb-Scargle of Differences')
    ax_diff_lomb.grid(True, linestyle='--', alpha=0.6)
    ax_diff_lomb.set_xscale('log')
    ax_diff_lomb.set_xlim(2, 500)

    # Ensure x-axis labels are visible and properly sized
    ax_diff_lomb.tick_params(axis='x', labelsize=9, pad=5)
    ax_diff_lomb.tick_params(axis='y', labelsize=9)

    # Add more x-tick labels to better show the log scale
    ax_diff_lomb.set_xticks([2, 5, 10, 20, 50, 100, 500, 1000])
    ax_diff_lomb.set_xticklabels(['2', '5', '10', '20', '50', '100', '500', '1000'])

    # Highlight important periods with vertical lines
    for period_val in [365.25, 182.625]:  # Annual and semi-annual
        ax_diff_lomb.axvline(x=period_val, color='k', linestyle='--', alpha=0.5)

    # Histogram subplot (bottom left)
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_hist.hist(differences.dropna(), bins=20, alpha=0.7, color='purple')
    ax_hist.axvline(x=0, color='k', linestyle='--')
    ax_hist.axvline(x=stats['mean'], color='r', linestyle='-', label=f'Mean: {stats["mean"]:.4f}')
    ax_hist.axvline(x=stats['median'], color='g', linestyle='-', label=f'Median: {stats["median"]:.4f}')
    ax_hist.set_xlabel(f'Difference (df-{reduce_components_name} - {compare_components_name}) for dU')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Distribution of Differences')
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, linestyle='--', alpha=0.6)

    # Correlation scatter plot (bottom middle)
    ax_scatter = fig.add_subplot(gs[2, 1])
    ax_scatter.scatter(df_common['dU'], comp_common['dU'], alpha=0.6, s=1, color='purple')
    # Add a diagonal line (perfect correlation)
    min_val = min(df_common['dU'].min(), comp_common['dU'].min())
    max_val = max(df_common['dU'].max(), comp_common['dU'].max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--')
    ax_scatter.set_xlabel(f'df-{reduce_components_name} dU values')
    ax_scatter.set_ylabel(f'{compare_components_name} dU values')
    ax_scatter.set_title(f'Correlation (r={stats["correlation"]:.4f}, R²={stats["r_squared"]:.4f})')
    ax_scatter.grid(True, linestyle='--', alpha=0.6)

    # Add a main title for the entire figure
    fig.suptitle(
        f'Comparison of dU Time Series (df-{reduce_components_name} vs {compare_components_name}) for Station {sta}\nSampling: {sampling}',
        fontsize=16, y=0.98)

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.25, wspace=0.24)

    return fig


def main():
    """Main function to process all stations with enhanced options for component handling."""
    # Parameters to customize analysis
    sampling = '01D'
    solution = 'ITRF2020-IGS-RES'
    stations = ['GOPE']  # Leave empty list to process all available stations

    # Define filter parameters
    # Example of a bandpass configuration
    filter_params = {
        'apply_filter': True,
        'lowcut': 1.0 / (400 * 24 * 60 * 60),
        'highcut': 1.0 / (30 * 24 * 60 * 60),
        'order': 3
    }

    # Example of a low-pass configuration
    # filter_params = {
    #     'apply_filter': True,
    #     'highcut': 1.0 / (400 * 24 * 60 * 60),
    #     'order': 3
    # }

    # Example of a high-pass configuration
    # filter_params = {
    #     'apply_filter': True,
    #     'lowcut': 1.0 / (30 * 24 * 60 * 60),
    #     'order': 3
    # }

    # Select which components to remove from the original data (set value to True to remove)
    reduce_components = {
        'A': 1,  # Atmospheric loading
        'O': 0,  # Ocean loading
        'S': 0,  # Surface water loading
        'H': 0  # Hydrological loading
    }

    # Select which components to include in the comparison sum (set value to True to include)
    compare_components = {
        'A': 0,  # Atmospheric loading
        'O': 0,  # Ocean loading
        'S': 0,  # Surface water loading
        'H': 0,  # Hydrological loading (old LSDM)
        'L': 1,  # Hydrological loading (Lisflood)
        'M': 0,  # Hydrological loading (LSDM)
    }

    # Date range for analysis
    start_date = datetime.date(2000, 1, 1)  # Set to None to use all available data
    end_date = datetime.date(2020, 12, 31)  # Set to None to use all available data

    # Find all available stations if none specified
    if len(stations) == 0:
        stations = find_stations(solution, sampling)
        with print_lock:
            print(f"Found {len(stations)} stations: {', '.join(stations)}")

    # Prepare output directory
    reduce_str = "".join([k for k, v in reduce_components.items() if v])
    compare_str = "".join([k for k, v in compare_components.items() if v])
    out_path = f"OUTPUT/SNX_LOAD_COMPARISONS/{solution}_{sampling}/PKL"
    with file_lock:
        os.makedirs(out_path, exist_ok=True)

    start_time = time.time()

    # Determine filter type and create appropriate filter string
    filter_str = ""
    with print_lock:
        print(f"\nStarting processing")
        print(f"Reducing original data by components: {reduce_str if reduce_str else 'None'}")
        print(f"Comparing with summed components: {compare_str if compare_str else 'None'}")

        if filter_params['apply_filter']:
            lowcut = filter_params.get('lowcut')
            highcut = filter_params.get('highcut')

            if lowcut is not None and highcut is not None:
                # Band-pass filter
                low_period = int(1 / lowcut / 86400)
                high_period = int(1 / highcut / 86400)
                print(f"Applying band-pass filter: {high_period}-{low_period} days (order {filter_params['order']})")
                filter_str = f"_BP_{high_period}d_{low_period}d"
            elif lowcut is not None and highcut is None:
                # High-pass filter
                low_period = int(1 / lowcut / 86400)
                print(f"Applying high-pass filter: >{low_period} days (order {filter_params['order']})")
                filter_str = f"_HP_{low_period}d"
            elif lowcut is None and highcut is not None:
                # Low-pass filter
                high_period = int(1 / highcut / 86400)
                print(f"Applying low-pass filter: <{high_period} days (order {filter_params['order']})")
                filter_str = f"_LP_{high_period}d"
            else:
                # No filtering
                print("No filter parameters specified, skipping filtering")
                filter_str = ""
        else:
            filter_str = ""

    # Process stations sequentially
    results = {}
    total = len(stations)

    for i, sta in enumerate(stations):
        try:
            with print_lock:
                print(f"\nProcessing station {i + 1}/{total}: {sta}")

            # Process the station
            result = process_station(
                sta, sampling, solution, reduce_components, compare_components,
                filter_params, start_date, end_date
            )

            if result is not None:
                results[sta] = result

            # Update progress
            with print_lock:
                print(f"Progress: {i + 1}/{total} stations completed ({(i + 1) / total:.1%})")

        except Exception as exc:
            with print_lock:
                print(f"Station {sta} generated an exception: {exc}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    with print_lock:
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        print(f"Successfully processed {len(results)} out of {len(stations)} stations.")

    # Create a summary of all stations
    if filter_params['apply_filter']:
        summary_file = f'{out_path}/{solution}_ALL_STATIONS_WO-{reduce_str}_VS_SUM-{compare_str}{filter_str}_SUMMARY.PKL'
    else:
        summary_file = f'{out_path}/{solution}_ALL_STATIONS_WO-{reduce_str}_VS_SUM-{compare_str}_SUMMARY.PKL'

    # Extract key statistics for all stations
    summary_stats = {}
    for sta, result in results.items():
        summary_stats[sta] = {
            'correlation': result['stats']['correlation'],
            'variance_explained': result['stats']['variance_explained'],
            'rms': result['stats']['rms'],
            'reduction_percent': result['reduction_percent'],
            'num_points': result['stats']['num_points'],
            'filter_params': result.get('filter_params'),
            'std': {
                'original': result['standard_deviations']['original_df'],
                'reduced': result['standard_deviations']['df_reduced'],
                'comparison': result['standard_deviations']['comparison_component'],
                'components': result['standard_deviations']['individual_components']
            }
        }

    with file_lock:
        pd.to_pickle(summary_stats, summary_file)

    with print_lock:
        print(f"\nSummary statistics saved to {summary_file}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()


