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
    export_coefficients
)
from toolbox_gravity_validation import (
    validate_coefficient_solution,
    analyze_coefficient_spectrum,
    plot_residual_map,
    plot_displacement_comparison_map,
    plot_vector_displacement_map,
    plot_displacement_components,
    analyze_temporal_evolution
)

def process_date(date, df, lat_lon, love_number_file,
                 max_degree, output_dir, frame,
                 calculate_errors=False,
                 monte_carlo=False,
                 print_maps=False,
                 regularization=False,
                 add_helmert=False):
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

    Returns:
    --------
    coeffs : dict
        Dictionary containing the calculated coefficients and errors
    """
    try:
        # Merge with station coordinates
        if 'Latitude' not in df.columns:
            merged_df = df.merge(lat_lon[['Latitude', 'Longitude']], left_on='CODE', right_index=True)
        else:
            merged_df = df.copy()
            del df  # Delete the original dataframe to free memory

        displacements = prepare_displacements_from_df(merged_df)
        # pd.DataFrame(displacements).to_csv('test_displacements_grid_IGS.csv')

        # Create date-specific output directory
        date_str = date.strftime('%Y%m%d')
        date_output_dir = os.path.join(output_dir, date_str)
        os.makedirs(date_output_dir, exist_ok=True)

        print(f"\nProcessing date: {date}")
        identifier = f'SLD_{max_degree}_{date_str}'
        # Process the dataframe to compute spherical harmonic coefficients
        coeffs = compute_gravity_field_coefficients(
            displacements=displacements,
            max_degree=max_degree,
            love_numbers_file=love_number_file,
            calculate_errors=calculate_errors,
            reference_frame=frame,
            save_summary=True,
            output_dir=date_output_dir,
            identifier=identifier,
            regularization=regularization,
            add_helmert=add_helmert
        )


        # Export the coefficients
        export_coefficients(coeffs, date_output_dir, prefix="gravity_coeffs",
                            identifier=identifier, icgem_format=True)

        # Create specific validation directory
        validation_dir = os.path.join(date_output_dir, "validation")
        os.makedirs(validation_dir, exist_ok=True)

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
            plot_displacement_components(displacements, save_path=os.path.join(validation_dir, f"{identifier}_displacements_map.png"), title=f"Displacements for {date_str}")
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

def main():
    """
    Main function to process all dates in the dataset with error analysis.
    """
    import argparse
    import gc
    from glob import glob
    default_solution = ''
    # default_solution = 'ESMGFZ_H_cf_GRIDS'
    # default_solution = 'ESMGFZ_H_cf_IGSNET'
    default_solution = 'IGS1R03SNX_01D'

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process displacement data with error analysis')
    parser.add_argument('--solution', type=str, default=default_solution,
                        help='Path to station coordinates file')
    parser.add_argument('--input', type=str, default=rf'INPUT_CRD/{default_solution}/TIME/',
                        help='Path to input data file')
    parser.add_argument('--sampling', type=str, default='01D',
                        help='Path to station coordinates file')
    parser.add_argument('--latlon', type=str, default='EXT/PROCESSINS_SUPPLEMENTS/ALL_STATIONS_LATLON.pkl',
                        help='Path to station coordinates file')
    parser.add_argument('--sta_availability', type=str, default='EXT/PROCESSINS_SUPPLEMENTS/ALL_STATIONS_AVAILABILITY.pkl',
                        help='Path to station coordinates file')
    parser.add_argument('--love', type=str, default=r'EXT/LLNs/ak135-LLNs-complete.dat',
                        help='Path to Love numbers file')
    parser.add_argument('--output', type=str, default=f'OUTPUT_NEW/{default_solution}',
                        help='Directory to save output files')
    parser.add_argument('--max-degree', type=int, default=20,
                        help='Maximum spherical harmonic degree')
    parser.add_argument('--frame', type=str, default='CF',
                        help='Displacement frame')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)', default='20180101')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)', default='20180101')
    parser.add_argument('--limit_stations', action='store_true', default=False,
                        help='Use only datum stations')
    parser.add_argument('--only_datum', action='store_true', default=False,
                        help='Use only datum stations')
    parser.add_argument('--errors', action='store_true', default=True,
                        help='Calculate formal errors')
    parser.add_argument('--printmaps', action='store_true', default=True,
                        help='Print Maps')
    parser.add_argument('--regularization', action='store_true', default=False,
                        help='Use regularization')

    args = parser.parse_args()

    args.input = rf'DATA/DISPLACEMENTS/{args.solution}/TIME/'
    args.output = f'OUTPUT/{args.solution}'

    print(f"Loading apriori station coordinates from {args.latlon}...")
    lat_lon = pd.read_pickle(args.latlon)
    print(f"Loading station availability from {args.sta_availability}...")
    station_availability = pd.read_pickle(args.sta_availability)
    station_availability.index = station_availability.index.date

    if args.limit_stations:
        args.output += '_LIM'

    if args.regularization:
        args.output += '_REG'

    args.output += '_TEST'

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

    files = glob(os.path.join(args.input,'*'))

    dates_to_process = filter_files_by_date_range(files,args.start_date,args.end_date)

    print(f"Processing {len(dates_to_process)} dates with max degree {args.max_degree}")

    # Process each date
    processed_dates = []
    for idx, filename in enumerate(dates_to_process):
        date = pd.to_datetime(os.path.basename(filename).split('_')[1],format='%Y%m%d')
        date_df = pd.read_pickle(filename)
        len_df = len(date_df)

        if args.limit_stations:
            stations_for_date = station_availability.loc[date.date()].dropna()
            len_df_snx = len(stations_for_date)
            if args.only_datum:
                stations_for_date = stations_for_date[stations_for_date == 1]

            len_df_snx_2 = len(stations_for_date)

            print(f'Number of stations: {len_df} -> {len_df_snx} -> {len_df_snx_2}')

            date_df = date_df.loc[date_df.index.get_level_values('CODE').isin(stations_for_date.index)]
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