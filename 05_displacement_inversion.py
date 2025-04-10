#!/usr/bin/env python

import matplotlib

matplotlib.use('TkAgg')

import pandas as pd
import os
import datetime
import gc  # Add garbage collection module

from toolbox_displacement_to_gravity_coefficients import (
    compute_gravity_field_coefficients,
    prepare_displacements_from_df,
    export_coefficients,
    save_processing_summary
)

from toolbox_gravity_validation import (
    validate_coefficient_solution,
    analyze_coefficient_spectrum,
    plot_residual_map,
    plot_displacement_comparison_map,
    plot_vector_displacement_map,
    plot_station_weights,
    plot_displacement_components,
    analyze_temporal_evolution
)


def process_date(date, df, lat_lon, love_number_file,
                 max_degree, output_dir, frame,
                 calculate_errors=False,
                 monte_carlo=False,
                 print_maps=False,
                 regularization=False,
                 add_helmert=False,
                 reduce_components=None,
                 use_vce=False):
    """
    Process a single date from the dataset.

    Parameters:
    -----------
    date : datetime
        Date to process
    df : pandas.DataFrame
        DataFrame containing data for this date
    lat_lon : pandas.DataFrame
        DataFrame containing latitude and longitude for all stations
    love_number_file : str
        Path to the Love numbers file
    max_degree : int
        Maximum spherical harmonic degree
    output_dir : str
        Directory to save output files
    calculate_errors : bool
        Whether to calculate formal errors
    monte_carlo : bool
        Whether to perform Monte Carlo analysis
    reduce_components : dict, optional
        Dictionary indicating which components to remove (A, O, S, H)

    Returns:
    --------
    coeffs : dict
        Dictionary containing the calculated coefficients and errors
    """
    try:
        # Convert date to string for file paths
        date_str = date.strftime('%Y%m%d')

        # Merge with station coordinates
        if 'Latitude' not in df.columns:
            merged_df = df.merge(lat_lon[['Latitude', 'Longitude']], left_on='CODE', right_index=True)
        else:
            merged_df = df.copy()
            del df  # Delete the original dataframe to free memory

        # Apply component reduction if specified
        components_name = "None"
        rms_reduction_info = None
        if reduce_components:
            merged_df, components_name, rms_reduction_info = reduce_components_from_data(merged_df, date_str, reduce_components)
            print(f"Reduced components: {components_name}")

        displacements = prepare_displacements_from_df(merged_df)

        # Create date-specific output directory with component reduction info if applicable
        date_output_dir = os.path.join(output_dir, date_str)

        os.makedirs(date_output_dir, exist_ok=True)

        print(f"\nProcessing date: {date}")

        # Add component reduction info to the identifier if applicable
        comp_suffix = f"_WO-{components_name}" if components_name != "None" else ""
        identifier = f'SLD_{max_degree}_{date_str}{comp_suffix}'

        # Process the dataframe to compute spherical harmonic coefficients
        coeffs = compute_gravity_field_coefficients(
            displacements=displacements,
            max_degree=max_degree,
            love_numbers_file=love_number_file,
            calculate_errors=calculate_errors,
            reference_frame=frame,
            regularization=regularization,
            add_helmert=add_helmert,
            use_vce=use_vce
        )

        # Add reference frame to the result dictionary for inclusion in summary
        coeffs['reference_frame'] = frame
        coeffs['max_degree'] = max_degree

        # Add RMS reduction information to coefficients if available
        if rms_reduction_info:
            coeffs['rms_reduction'] = rms_reduction_info

        # Save the summary
        summary_files = save_processing_summary(
            coeffs,
            output_dir=date_output_dir,
            identifier=identifier,
            formats=['yaml']
        )

        # Export the coefficients
        export_coefficients(coeffs, date_output_dir, prefix="gravity_coeffs",
                            identifier=identifier, icgem_format=True)

        # Create specific validation directory
        validation_dir = os.path.join(date_output_dir, "validation")
        os.makedirs(validation_dir, exist_ok=True)

        if use_vce:
            plot_station_weights(displacements['lat'],
                                 displacements['lon'],
                                 coeffs['vce_weights'],
                                 displacements['code'],
                                 validation_dir+f'/{identifier}_vce.png')

        # Validate solution if not already done during coefficient computation
        if 'validation' not in coeffs:
            validation = validate_coefficient_solution(
                displacements,
                coeffs,
                love_number_file=love_number_file,
                reference_frame=frame
            )
            coeffs['validation'] = validation
        else:
            validation = coeffs['validation']

        if print_maps:
            # Create residual maps
            plot_displacement_components(displacements,
                                         save_path=os.path.join(validation_dir, f"{identifier}_displacements_map.png"),
                                         title=f"Displacements for {date_str}")
            print("Creating residual maps...")
            residual_map = plot_residual_map(
                validation,
                save_path=os.path.join(validation_dir, f"{identifier}_residual_map.png"),
                title=f"Residual Distribution for {date_str} (max_degree={max_degree})"
            )
            del residual_map  # Free memory

            # Create displacement comparison maps for each component
            for component in ['vertical']:
                comparison_map = plot_displacement_comparison_map(
                    displacements,
                    validation['reconstructed'],
                    component=component,
                    save_path=os.path.join(validation_dir, f"{identifier}_{component}_comparison.png"),
                    title=f"{component.capitalize()} Displacement Comparison for {date_str}"
                )
                del comparison_map  # Free memory

        # Analyze coefficient spectrum if not already done
        if 'spectral_analysis' not in coeffs:
            spectral_results = analyze_coefficient_spectrum(coeffs, max_degree)
            coeffs['spectral_analysis'] = spectral_results

            # Save spectrum plot
            spectral_results['figure'].savefig(
                os.path.join(validation_dir, f"{identifier}_spectrum.png"),
                dpi=300, bbox_inches='tight'
            )
            # Close the figure to free memory
            import matplotlib.pyplot as plt
            plt.close(spectral_results['figure'])

        # Print validation results
        print(f"Validation results:")
        print(f"  Variance explained: North={validation['variance_explained']['north']:.2%}, "
              f"East={validation['variance_explained']['east']:.2%}, "
              f"Up={validation['variance_explained']['up']:.2%}")
        print(f"  RMS errors: North={validation['rms']['north']:.3e}m, "
              f"East={validation['rms']['east']:.3e}m, "
              f"Up={validation['rms']['up']:.3e}m")

        # Print summary information
        print(f"Processed {date_str}:")
        print(f"  - Load coefficients - max degree: {coeffs['load_coefficients'].coeffs.shape[1] - 1}")
        print(f"  - Potential coefficients - max degree: {coeffs['potential_coefficients'].coeffs.shape[1] - 1}")

        if 'residuals' in coeffs and coeffs['residuals'] is not None:
            print(f"  - RMS of residuals: {coeffs['residuals']:.6e} m")

        # Add component reduction info to the summary if applicable
        if components_name != "None":
            print(f"  - Components removed: {components_name}")

        print(f"  - Results saved to: {date_output_dir}")

        return coeffs

    except Exception as e:
        print(f"Error in process_date for {date}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up local variables to ensure memory is released
        if 'merged_df' in locals():
            del merged_df
        if 'displacements' in locals():
            del displacements
        if 'validation' in locals():
            del validation
        if 'spectral_results' in locals():
            del spectral_results
        if 'rms_reduction_info' in locals():
            del rms_reduction_info

        # Force garbage collection
        gc.collect()
def filter_files_by_date_range(directory_path, start_date, end_date):
    """
    Filter files in a directory based on date range in filename.

    Parameters:
    directory_path (str): Path to the directory containing the files
    start_date (str): Start date in format 'YYYYMMDD'
    end_date (str): End date in format 'YYYYMMDD'

    Returns:
    list: List of filenames that fall within the date range
    """
    # Convert string dates to datetime objects for comparison
    start_date_obj = pd.to_datetime(start_date, format='%Y%m%d')
    end_date_obj = pd.to_datetime(end_date, format='%Y%m%d')

    filtered_files = []
    # List all files in the directory
    for filename in directory_path:
        # Extract date from filename
        file_date_str = os.path.basename(filename).split('_')[1]
        file_date_obj = pd.to_datetime(file_date_str, format = '%Y%m%d')

        # Check if the file date is within the specified range
        if start_date_obj <= file_date_obj <= end_date_obj:
            filtered_files.append(filename)

    return filtered_files


def subtract_dataframes(original_df, combined_df):
    """
    Subtract values from combined_df from original_df, aligning by EPOCH and CODE.

    Parameters:
    -----------
    original_df : pandas.DataFrame
        Original dataframe with MultiIndex (EPOCH, CODE, SOLN, DOMES)
    combined_df : pandas.DataFrame
        Dataframe with values to subtract, index should contain EPOCH and CODE

    Returns:
    --------
    pandas.DataFrame
        Resulting dataframe with the same structure as original_df
        but with dN, dE, dU values subtracted
    """
    # Create copies to avoid modifying the originals
    df1 = original_df.copy()
    df2 = combined_df.copy()

    # Reset index of the original dataframe to work with it more easily
    df1_reset = df1.reset_index()

    # If the combined_df is not already reset, reset it too
    if isinstance(df2.index, pd.MultiIndex) or df2.index.name:
        df2_reset = df2.reset_index()
    else:
        df2_reset = df2.copy()

    # Extract EPOCH and CODE from combined_df index if they're in the index
    # If the combined_df has EPOCH and CODE as columns already, skip this part
    if 'EPOCH' not in df2_reset.columns and 'index' in df2_reset.columns:
        # Parse the index assuming format like "2018-01-02/00NA"
        df2_reset[['EPOCH', 'CODE']] = df2_reset['index'].str.split('/', n=1, expand=True)
        df2_reset = df2_reset.drop('index', axis=1)

    # Now merge the dataframes on EPOCH and CODE
    result = pd.merge(df1_reset, df2_reset[['EPOCH', 'CODE', 'dN', 'dE', 'dU']],
                      on=['EPOCH', 'CODE'],
                      how='left',
                      suffixes=('', '_to_subtract'))

    # Perform the subtraction for rows that have matching values in combined_df
    # Handle NaN values that may occur if there's no match in combined_df
    result['dN'] = result.apply(lambda row: row['dN'] - row['dN_to_subtract']
    if pd.notna(row['dN_to_subtract']) else row['dN'], axis=1)
    result['dE'] = result.apply(lambda row: row['dE'] - row['dE_to_subtract']
    if pd.notna(row['dE_to_subtract']) else row['dE'], axis=1)
    result['dU'] = result.apply(lambda row: row['dU'] - row['dU_to_subtract']
    if pd.notna(row['dU_to_subtract']) else row['dU'], axis=1)

    # Drop the columns we no longer need
    result = result.drop(['dN_to_subtract', 'dE_to_subtract', 'dU_to_subtract'], axis=1)

    # Set the index back to original form
    index_columns = [x for x in result.columns if x in ['EPOCH', 'CODE', 'SOLN', 'DOMES','FLAG','S']]
    result = result.set_index(index_columns)

    return result


def calculate_rms(df):
    """
    Calculate RMS values for displacement components.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing dN, dE, dU columns

    Returns:
    --------
    dict
        Dictionary with RMS values for each component
    """
    import numpy as np

    # Calculate RMS for each component
    rms_n = np.sqrt(np.mean(df['dN'] ** 2))
    rms_e = np.sqrt(np.mean(df['dE'] ** 2))
    rms_u = np.sqrt(np.mean(df['dU'] ** 2))

    return {
        'north': float(rms_n),
        'east': float(rms_e),
        'up': float(rms_u)
    }


def reduce_components_from_data(df, date_str, reduce_components):
    """
    Remove selected components from the original data and track RMS reduction.

    Parameters:
    -----------
    df : pandas.DataFrame
        Original displacement data
    date_str : str
        Date string in format 'YYYYMMDD'
    reduce_components : dict
        Dictionary indicating which components to remove

    Returns:
    --------
    tuple
        (reduced_df, components_name, rms_reduction_info) where:
        - reduced_df is the DataFrame with components removed
        - components_name is a string representation of the removed components
        - rms_reduction_info is a dictionary with RMS reduction information
    """
    import pandas as pd
    import os
    import datetime

    # Create list of components to remove
    component_labels = []
    component_dfs = []

    # Calculate original RMS values
    original_rms = calculate_rms(df)
    print(
        f"Original RMS: North={original_rms['north']:.3e}m, East={original_rms['east']:.3e}m, Up={original_rms['up']:.3e}m")

    # Store RMS reduction information
    rms_reduction_info = {
        'original_rms': original_rms,
        'component_reductions': {}
    }

    # Determine which components to reduce
    components_to_process = []

    if reduce_components['A']:
        file_path = f'DATA/DISPLACEMENTS/ESMGFZ_A_cf_IGSNET/TIME/ESMGFZ_{date_str}_A_cf_DISP.PKL'
        if os.path.exists(file_path):
            components_to_process.append(('A', file_path))
            print(f"Found component A data for {date_str}")

    if reduce_components['O']:
        file_path = f'DATA/DISPLACEMENTS/ESMGFZ_O_cf_IGSNET/TIME/ESMGFZ_{date_str}_O_cf_DISP.PKL'
        if os.path.exists(file_path):
            components_to_process.append(('O', file_path))
            print(f"Found component O data for {date_str}")

    if reduce_components['S']:
        file_path = f'DATA/DISPLACEMENTS/ESMGFZ_S_cf_IGSNET/TIME/ESMGFZ_{date_str}_S_cf_DISP.PKL'
        if os.path.exists(file_path):
            components_to_process.append(('S', file_path))
            print(f"Found component S data for {date_str}")

    if reduce_components['H']:
        file_path = f'DATA/DISPLACEMENTS/ESMGFZ_H_cf_IGSNET/TIME/ESMGFZ_{date_str}_H_cf_DISP.PKL'
        if os.path.exists(file_path):
            components_to_process.append(('H', file_path))
            print(f"Found component H data for {date_str}")

    # If no components to reduce, return original data
    if not components_to_process:
        return df, "None", rms_reduction_info

    # Function to truncate datetime to date in MultiIndex
    def truncate_date_in_multiindex(df):
        # Create a copy to avoid modifying the original
        df_copy = df.copy()

        # Check if we need to truncate EPOCH to date
        if isinstance(df.index, pd.MultiIndex) and 'EPOCH' in df.index.names:
            # Get the current index values
            index_values = list(df.index)
            epoch_pos = df.index.names.index('EPOCH')

            # Create new index with truncated datetime
            new_index = []
            for idx in index_values:
                idx_list = list(idx)
                # Truncate datetime to date
                if isinstance(idx[epoch_pos], (pd.Timestamp, datetime.datetime)):
                    idx_list[epoch_pos] = idx[epoch_pos].date()
                new_index.append(tuple(idx_list))

            # Set the new index
            df_copy.index = pd.MultiIndex.from_tuples(new_index, names=df.index.names)

        return df_copy

    # Create a copy of the original DataFrame with truncated dates
    df_red = df.copy()
    df_date_trunc = truncate_date_in_multiindex(df)

    # Process each component dataframe sequentially and track RMS reduction
    current_df = df_date_trunc.copy()

    for comp_label, file_path in components_to_process:
        # Load component data
        comp_df = pd.read_pickle(file_path)
        comp_df = truncate_date_in_multiindex(comp_df)

        # Subtract this component
        next_df = subtract_dataframes(current_df, comp_df)

        # Calculate RMS after this component reduction
        reduced_rms = calculate_rms(next_df)

        # Calculate percentage reduction
        percent_reduction = {
            'north': float((original_rms['north'] - reduced_rms['north']) / original_rms['north'] * 100),
            'east': float((original_rms['east'] - reduced_rms['east']) / original_rms['east'] * 100),
            'up': float((original_rms['up'] - reduced_rms['up']) / original_rms['up'] * 100)
        }

        # Print RMS reduction information
        print(f"After removing component {comp_label}:")
        print(f"  RMS: North={reduced_rms['north']:.3e}m, East={reduced_rms['east']:.3e}m, Up={reduced_rms['up']:.3e}m")
        print(
            f"  Reduction: North={percent_reduction['north']:.2f}%, East={percent_reduction['east']:.2f}%, Up={percent_reduction['up']:.2f}%")

        # Store RMS reduction information
        rms_reduction_info['component_reductions'][comp_label] = {
            'rms': reduced_rms,
            'percent_reduction': percent_reduction
        }

        # Update current dataframe for next iteration
        current_df = next_df
        component_labels.append(comp_label)

    # Final reduced dataframe
    df_red = current_df

    # Calculate total RMS reduction
    final_rms = calculate_rms(df_red)
    total_reduction = {
        'north': (original_rms['north'] - final_rms['north']) / original_rms['north'] * 100,
        'east': (original_rms['east'] - final_rms['east']) / original_rms['east'] * 100,
        'up': (original_rms['up'] - final_rms['up']) / original_rms['up'] * 100
    }

    # Store total reduction information
    rms_reduction_info['final_rms'] = final_rms
    rms_reduction_info['total_reduction'] = total_reduction

    # Print total reduction information
    print(f"Total RMS reduction after removing {', '.join(component_labels)}:")
    print(f"  Final RMS: North={final_rms['north']:.3e}m, East={final_rms['east']:.3e}m, Up={final_rms['up']:.3e}m")
    print(
        f"  Total reduction: North={total_reduction['north']:.2f}%, East={total_reduction['east']:.2f}%, Up={total_reduction['up']:.2f}%")

    # Create a string representing the removed components
    components_name = ''.join(component_labels)

    return df_red, components_name, rms_reduction_info


def main():
    """
    Main function to process all dates in the dataset with error analysis.
    """
    import argparse
    import gc
    from glob import glob
    default_solution = ''
    # default_solution = 'ESMGFZ_LSDM_cf_GRIDS'
    # default_solution = 'ESMGFZ_H_cf_IGSNET'
    # default_solution = 'IGS1R03SNX_01D'
    default_solution = 'ITRF2020-IGS-RES_01D'

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process displacement data with error analysis')
    parser.add_argument('--solution', type=str, default=default_solution,
                        help='Path to station coordinates file')
    parser.add_argument('--sampling', type=str, default='01D',
                        help='Path to station coordinates file')
    parser.add_argument('--latlon', type=str, default='EXT/PROCESSINS_SUPPLEMENTS/ALL_STATIONS_LATLON.pkl',
                        help='Path to station coordinates file')
    parser.add_argument('--sta_availability', type=str,
                        default='EXT/PROCESSINS_SUPPLEMENTS/ALL_STATIONS_AVAILABILITY.pkl',
                        help='Path to station coordinates file')
    parser.add_argument('--love', type=str, default=r'EXT/LLNs/ak135-LLNs-complete.dat',
                        help='Path to Love numbers file')
    parser.add_argument('--max-degree', type=int, default=7,
                        help='Maximum spherical harmonic degree')
    parser.add_argument('--frame', type=str, default='CF',
                        help='Displacement frame')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)', default='20180102')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)', default='20180102')
    parser.add_argument('--limit_stations', action='store_true', default=False,
                        help='Use only datum stations')
    parser.add_argument('--only_datum', action='store_true', default=False,
                        help='Use only datum stations')
    parser.add_argument('--use_vce', action='store_true', default=False,
                        help='Use VCE for adding error information') ## Doesnt work
    parser.add_argument('--errors', action='store_true', default=True,
                        help='Calculate formal errors')
    parser.add_argument('--printmaps', action='store_true', default=True,
                        help='Print Maps')
    parser.add_argument('--regularization', action='store_true', default=False,
                        help='Use regularization')

    # Add arguments for component reduction
    parser.add_argument('--reduce_A', action='store_true', default=0,
                        help='Reduce atmospheric loading component')
    parser.add_argument('--reduce_O', action='store_true', default=0,
                        help='Reduce ocean loading component')
    parser.add_argument('--reduce_S', action='store_true', default=0,
                        help='Reduce surface water loading component')
    parser.add_argument('--reduce_H', action='store_true', default=0,
                        help='Reduce hydrological loading component')

    args = parser.parse_args()

    args.input = os.path.join('DATA','DISPLACEMENTS',args.solution,'TIME')
    args.output = os.path.join('OUTPUT',args.solution)

    print(f"Loading apriori station coordinates from {args.latlon}...")
    lat_lon = pd.read_pickle(args.latlon)

    if args.limit_stations:
        args.output += '_LIM'

    if args.regularization:
        args.output += '_REG'

    # Create component reduction dictionary
    reduce_components = None
    if any([args.reduce_A, args.reduce_O, args.reduce_S, args.reduce_H]):
        reduce_components = {
            'A': args.reduce_A,
            'O': args.reduce_O,
            'S': args.reduce_S,
            'H': args.reduce_H
        }

        # Add component reduction info to output path
        component_str = ''
        if args.reduce_A:
            component_str += 'A'
        if args.reduce_O:
            component_str += 'O'
        if args.reduce_S:
            component_str += 'S'
        if args.reduce_H:
            component_str += 'H'

        if component_str:
            args.output += f'_WO-{component_str}'

    # args.output += '_TEST'

    output = os.path.join(args.output, os.path.basename(args.input).split('.')[0])
    # Create output directory
    os.makedirs(output, exist_ok=True)

    # Parse date range if provided
    start_date = None
    end_date = None

    if args.start_date:
        start_date = datetime.datetime.strptime(args.start_date, '%Y%m%d')
        print(f"Starting from {start_date}")

    if args.end_date:
        end_date = datetime.datetime.strptime(args.end_date, '%Y%m%d')
        print(f"Ending at {end_date}")

    print(args.input)
    files = glob(os.path.join(args.input, '*'))

    dates_to_process = filter_files_by_date_range(files, args.start_date, args.end_date)

    print(f"Processing {len(dates_to_process)}/{len(files)} dates with max degree {args.max_degree}")

    # Print component reduction info if applicable
    if reduce_components:
        component_list = [k for k, v in reduce_components.items() if v]
        if component_list:
            print(f"Reducing components: {', '.join(component_list)}")

            # Check if component data files exist for the date range
            for component in component_list:
                if component in ['A', 'O', 'S', 'H']:
                    comp_dir = f'DATA/DISPLACEMENTS/ESMGFZ_{component}_cf_IGSNET/TIME/'
                    if not os.path.exists(comp_dir):
                        print(f"Warning: Component directory {comp_dir} does not exist.")
                    else:
                        # Check for some example files
                        example_files = glob(os.path.join(comp_dir, f'ESMGFZ_*_{component}_cf_DISP.PKL'))
                        if example_files:
                            print(f"Found {len(example_files)} files for component {component}")
                        else:
                            print(f"Warning: No files found for component {component} in {comp_dir}")

    # Process each date
    processed_dates = []
    for idx, filename in enumerate(dates_to_process):
        date = pd.to_datetime(os.path.basename(filename).split('_')[1], format='%Y%m%d')
        date_df = pd.read_pickle(filename)
        len_df = len(date_df)

        if args.limit_stations:
            try:
                print(f"Loading station availability from {args.sta_availability}...")
                station_availability = pd.read_pickle(args.sta_availability)
                station_availability.index = station_availability.index.date
                stations_for_date = station_availability.loc[date.date()].dropna()
                len_df_snx = len(stations_for_date)
                if args.only_datum:
                    stations_for_date = stations_for_date[stations_for_date == 1]

                len_df_snx_2 = len(stations_for_date)

                print(f'Number of stations: {len_df} -> {len_df_snx} -> {len_df_snx_2}')

                date_df = date_df.loc[date_df.index.get_level_values('CODE').isin(stations_for_date.index)]
            except:
                print('Failed to get station availability information')
        else:
            print(f'Number of stations: {len_df}')

        try:
            print(f"\nProcessing date {idx + 1}/{len(dates_to_process)}: {date}")

            # Process the date
            coeffs = process_date(
                date=date,
                df=date_df,
                lat_lon=lat_lon,
                love_number_file=args.love,
                max_degree=args.max_degree,
                output_dir=output,
                calculate_errors=args.errors,
                frame=args.frame,
                print_maps=args.printmaps,
                regularization=args.regularization,
                add_helmert=False,
                reduce_components=reduce_components,
                use_vce=args.use_vce
            )

            if coeffs is not None:
                processed_dates.append(date)

            # Explicitly clear memory
            del date_df
            del coeffs

            # Force garbage collection after each date
            gc.collect()

            # Print memory status (if psutil is available)
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                print(f"Memory usage after date {date}: {memory_info.rss / (1024 * 1024):.2f} MB")
            except ImportError:
                pass  # psutil not available, skip memory reporting

        except Exception as e:
            print(f"Error processing date {date}: {e}")
            import traceback
            traceback.print_exc()

            # Still need to clean up even if there's an error
            if 'date_df' in locals():
                del date_df
            gc.collect()

    print("\nProcessing complete!")
    print(f"Successfully processed {len(processed_dates)} out of {len(dates_to_process)} dates")

if __name__ == "__main__":
    main()