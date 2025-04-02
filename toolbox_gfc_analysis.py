import matplotlib
matplotlib.use('TkAgg')

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import pyshtools as pysh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy.signal import lombscargle
from matplotlib.ticker import LogLocator, LogFormatter
from gcc_analysis_tools import apply_lowpass_filter

def create_coefficient_time_series(base_dir='', solution='IGS1R03SNX_LOAD_CRD_CF_H_7D',
                                   coeff_type='potential', degree=1, order=0,
                                   start_date=None, end_date=None, degmax=6):
    """
    Create a time series of a specific spherical harmonic coefficient from GFC files
    using pyshtools to read ICGEM format.

    Parameters:
    -----------
    base_dir : str
        Base directory containing the results (e.g., 'OUTPUT_WITH_ERRORS')
    solution : str
        Solution directory name (e.g., 'IGS1R03SNX_LOAD_CRD_CM_H_7D')
    coeff_type : str
        Type of coefficient ('potential' or 'load')
    degree : int
        Degree of the coefficient to extract
    order : int
        Order of the coefficient to extract
    start_date : str, optional
        Start date in format 'YYYYMMDD' to filter results
    end_date : str, optional
        End date in format 'YYYYMMDD' to filter results

    Returns:
    --------
    dict :
        Dictionary containing dates, C coefficients, S coefficients and their errors
    """

    # Create full directory path
    solution_dir = os.path.join(base_dir, solution)

    # Find all date directories
    date_dirs = sorted(glob.glob(os.path.join(solution_dir, '[0-9]' * 8)))

    if not date_dirs:
        print(f"No date directories found in {solution_dir}")
        return None

    # Container for results
    dates = []
    date_strs = []
    c_values = []
    s_values = []
    c_errors = []
    s_errors = []

    # Filter by date range if specified
    for date_dir in date_dirs:
        date_str = os.path.basename(date_dir)

        # Skip if not a proper date format
        if not re.match(r'\d{8}', date_str):
            continue

        # Filter by date range if specified
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue

        # Look for GFC files
        gfc_pattern = f"*_{degmax}_*{coeff_type}*.gfc"
        gfc_files = glob.glob(os.path.join(date_dir, gfc_pattern))

        if len(gfc_files) == 0:
            continue
        gfc_file = gfc_files[0]

        # Parse the file using pyshtools
        try:
            # Load the coefficients
            try:
                clm = pysh.SHGravCoeffs.from_file(gfc_file, format='icgem',errors='formal',epoch=date_str.replace('-',''))
                # Extract the specific coefficient
                c_val = clm.coeffs[0, degree, order]
                s_val = clm.coeffs[1, degree, order] if order > 0 else 0.0

                # Extract errors if available
                try:
                    c_err = clm.errors[0, degree, order]
                    s_err = clm.errors[1, degree, order] if order > 0 else 0.0
                except (AttributeError, IndexError):
                    c_err = np.nan
                    s_err = np.nan

                # Convert date string to datetime for plotting
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                dates.append(date_obj)
                date_strs.append(date_str)
                c_values.append(c_val)
                s_values.append(s_val)
                c_errors.append(c_err)
                s_errors.append(s_err)

            except Exception as e:
                print(f"Error loading {gfc_file} with pyshtools: {e}")

        except Exception as e:
            print(f"Error processing {gfc_file}: {e}")

    if not dates:
        print("No valid data found for the specified parameters")
        return None

    # Convert to numpy arrays
    dates = np.array(dates)
    c_values = np.array(c_values)
    s_values = np.array(s_values)
    c_errors = np.array(c_errors)
    s_errors = np.array(s_errors)

    # Create DataFrame for easier analysis
    df = pd.DataFrame({
        'date': dates,
        'date_str': date_strs,
        'C': c_values,
        'S': s_values,
        'C_error': c_errors,
        'S_error': s_errors
    })

    # Sort by date
    df = df.sort_values('date').set_index('date')

    return df

def plot_coefficient_time_series(series_dict, x_column='date', y_columns=None, error_columns=None,
                                 title=None, xlabel='Date', ylabel='Coefficient Value',
                                 figsize=(12, 8), colors=None, markers=None, linestyles=None,
                                 save_path=None, dpi=300):
    """
    Plot multiple coefficient time series on the same plot from dataframes.

    Parameters:
    -----------
    series_dict : dict
        Dictionary where keys are series labels and values are dataframes containing
        the data to plot.
    x_column : str, optional
        Name of the column in each dataframe to use for x-axis values (typically dates).
    y_columns : dict or None, optional
        Dictionary mapping series labels to column names for y values.
        If None, will use column 'C' for each dataframe.
    error_columns : dict or None, optional
        Dictionary mapping series labels to column names for error values.
        If None, will look for '{y_column}_error' in each dataframe.
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    figsize : tuple, optional
        Figure size (width, height)
    colors : dict or None, optional
        Dictionary mapping series labels to colors
    markers : dict or None, optional
        Dictionary mapping series labels to marker styles
    linestyles : dict or None, optional
        Dictionary mapping series labels to line styles
    save_path : str, optional
        Path to save the plot. If None, the plot will not be saved.
    dpi : int, optional
        DPI for the saved figure

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)

    # Initialize optional parameter dictionaries if None
    y_columns = y_columns or {label: 'C' for label in series_dict}
    error_columns = error_columns or {label: f"{y_columns.get(label, 'C')}_error" for label in series_dict}
    colors = colors or {}
    markers = markers or {}
    linestyles = linestyles or {}

    for label, df in series_dict.items():
        # Get column names for this dataframe
        y_col = y_columns.get(label, 'C')
        error_col = error_columns.get(label, f"{y_col}_error")

        # Extract data
        x_data = df[x_column]
        y_data = df[y_col]

        # Set plot properties with defaults
        plot_kwargs = {
            'label': label,
            'color': colors.get(label, None),  # None will use default color cycle
            'marker': markers.get(label, 'o'),
            'linestyle': linestyles.get(label, '-')
        }

        # Remove None values to let matplotlib use defaults
        plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}

        # Plot with error bars if error column exists and contains non-NaN values
        if error_col in df.columns and np.any(~np.isnan(df[error_col])):
            ax.errorbar(x_data, y_data, yerr=df[error_col], capsize=3, **plot_kwargs)
        else:
            ax.plot(x_data, y_data, **plot_kwargs)

    # Set labels and grid
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    # Format x-axis date labels if x contains dates
    if pd.api.types.is_datetime64_any_dtype(x_data):
        fig.autofmt_xdate()

    # Adjust layout
    plt.tight_layout()

    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Time series plot saved to {save_path}")

    return fig, ax
    # plt.show()
    #
    # # Return data for further analysis
    # return {
    #     'dates': df['date'].values,
    #     'date_strs': df['date_str'].values,
    #     'C': df['C'].values,
    #     'S': df['S'].values,
    #     'C_error': df['C_error'].values,
    #     'S_error': df['S_error'].values,
    #     'dataframe': df
    # }


def plot_geocenter_motion(solutions_dict, coeff_columns=['X', 'Y', 'Z'],
                          low_pass=False,
                          error_suffix='_error',
                          figsize=(15, 12), save_path=None, dpi=300,
                          colors=None, markers=None, linestyles=None,
                          legend_loc='upper left', alpha=0.7, plot_errors=False,
                          min_period=10, max_period=1000, n_freqs=1000):
    """
    Plot geocenter motion from multiple solutions with dual layout:
    - Left: Time series of geocenter coordinates (2/3 width)
    - Right: Lomb-Scargle periodogram with amplitudes in mm (1/3 width)

    Parameters:
    -----------
    solutions_dict : dict
        Dictionary with solution names as keys and DataFrames as values,
        each containing geocenter coordinate columns (X, Y, Z) and their error columns
    coeff_columns : list, optional
        List of coordinate column names to plot (should be X, Y, Z)
    low_pass : bool, optional
        Whether to apply a low-pass filter to the time series
    error_suffix : str, optional
        Suffix used for error columns (e.g., 'X_error' if suffix is '_error')
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot. If None, the plot will not be saved.
    dpi : int, optional
        DPI for the saved figure
    colors : dict or None, optional
        Dictionary with solution names as keys and colors as values
    markers : dict or None, optional
        Dictionary with solution names as keys and markers as values
    linestyles : dict or None, optional
        Dictionary with solution names as keys and linestyles as values
    legend_loc : str, optional
        Location of the legend
    alpha : float, optional
        Alpha transparency for the plot markers and lines
    plot_errors : bool, optional
        Whether to plot error bars
    min_period : float, optional
        Minimum period (in days) for Lomb-Scargle analysis
    max_period : float, optional
        Maximum period (in days) for Lomb-Scargle analysis
    n_freqs : int, optional
        Number of frequency points to use in Lomb-Scargle analysis

    Returns:
    --------
    fig, time_axes, ls_axes : matplotlib figure and lists of time series and Lomb-Scargle axes
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.dates as mdates
    from scipy.signal import lombscargle

    # Create components labels
    # Using the column names directly since they already represent X, Y, Z coordinates
    components = coeff_columns

    # Number of subplots
    n_plots = len(coeff_columns)

    # Create figure with custom grid layout (2:1 ratio)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_plots, 3, figure=fig)

    # Lists to store axes
    time_axes = []
    ls_axes = []

    # Define default aesthetic dictionaries if not provided
    solution_names = list(solutions_dict.keys())
    n_solutions = len(solution_names)

    # Default color palette
    default_colors = plt.cm.tab10(np.linspace(0, 1, n_solutions))

    if colors is None:
        colors = {name: default_colors[i] for i, name in enumerate(solution_names)}
    elif isinstance(colors, list):
        colors = {name: colors[i] if i < len(colors) else default_colors[i]
                  for i, name in enumerate(solution_names)}

    if markers is None:
        markers = {name: None for i, name in enumerate(solution_names)}
    elif isinstance(markers, list):
        markers = {name: markers[i] if i < len(markers) else 'o'
                   for i, name in enumerate(solution_names)}

    if linestyles is None:
        linestyles = {name: '-' for i, name in enumerate(solution_names)}
    elif isinstance(linestyles, list):
        linestyles = {name: linestyles[i] if i < len(linestyles) else '-'
                      for i, name in enumerate(solution_names)}

    # Keep track of minimum and maximum y values for each coordinate
    y_mins = [np.inf] * n_plots
    y_maxs = [-np.inf] * n_plots

    # Keep track of maximum amplitude for periodograms (in mm)
    amp_maxs = [0] * n_plots

    # First pass: collect data to determine y-axis limits
    for i, coord in enumerate(coeff_columns):
        for sol_name, df in solutions_dict.items():
            if coord not in df.columns:
                continue

            y_data = df[coord].dropna()
            if len(y_data) > 0:
                y_mins[i] = min(y_mins[i], np.nanmin(y_data))
                y_maxs[i] = max(y_maxs[i], np.nanmax(y_data))

                # For periodogram amplitude limits
                if len(y_data) >= 5:
                    x_dates = y_data.index
                    x_days = np.array([(d - x_dates[0]).total_seconds() / (24 * 3600) for d in x_dates])
                    y_vals = y_data.values

                    periods = np.logspace(np.log10(min_period), np.log10(max_period), n_freqs)
                    freqs = 1 / periods

                    try:
                        pgram = lombscargle(x_days, y_vals - np.mean(y_vals), 2 * np.pi * freqs, normalize=True)

                        # Calculate amplitudes in mm (assuming input data is already in mm)
                        amplitudes = np.sqrt(4 * pgram / len(y_vals))
                        amp_maxs[i] = max(amp_maxs[i], np.max(amplitudes))
                    except Exception as e:
                        print(f"Error in first pass calculation for {sol_name}, {coord}: {e}")

    # Second pass: create plots
    for i, coord in enumerate(coeff_columns):
        # Create time series axes (2 columns wide)
        ax_time = fig.add_subplot(gs[i, 0:2])
        time_axes.append(ax_time)

        # Set title for time series plot
        ax_time.set_title(f'Geocenter {coord} Motion')

        # Plot each solution's time series
        for sol_name, df in solutions_dict.items():
            if coord not in df.columns:
                print(f"Warning: {coord} not found in solution {sol_name}, skipping.")
                continue

            # Original data and errors
            x_data = df.index
            y_data = df[coord]

            # Check for error columns
            error_col = f"{coord}{error_suffix}" if error_suffix else None
            has_errors = error_col in df.columns and np.any(~np.isnan(df[error_col]))
            if plot_errors is False:
                has_errors = False

            # Plot data
            if has_errors and plot_errors:
                # Make sure all error values are positive
                error_values = df[error_col].abs().copy()

                ax_time.errorbar(x_data, y_data, yerr=error_values,
                                 fmt=f"{markers[sol_name]}", linestyle=linestyles[sol_name],
                                 color=colors[sol_name], capsize=3, alpha=alpha,
                                 label=f"{sol_name}", linewidth=1)
            elif low_pass:
                # Apply lowpass filter if requested
                cutoff_period = 150  # days
                cutoff_frequency = 1 / cutoff_period
                sampling_rate = 1  # Data is sampled every 1 day

                # Ensure y_data doesn't have NaNs for filtering
                valid_indices = ~np.isnan(y_data.values)
                if np.sum(valid_indices) > 5:  # Need enough valid data points
                    temp_signal = y_data.values.copy()
                    # Replace NaNs with interpolated values for filtering
                    if np.any(~valid_indices):
                        temp_signal[~valid_indices] = np.interp(
                            np.flatnonzero(~valid_indices),
                            np.flatnonzero(valid_indices),
                            temp_signal[valid_indices]
                        )

                    lowpass_filtered_signal = apply_lowpass_filter(temp_signal, cutoff_frequency, sampling_rate)

                    # Only plot at valid data points
                    ax_time.plot(x_data[valid_indices], lowpass_filtered_signal[valid_indices],
                                 marker=markers[sol_name], linestyle=linestyles[sol_name],
                                 color=colors[sol_name], linewidth=2, label=f"{sol_name}")
                else:
                    # Not enough valid data points for filtering
                    ax_time.plot(x_data, y_data, marker=markers[sol_name],
                                 linestyle=linestyles[sol_name], color=colors[sol_name],
                                 linewidth=1, label=f"{sol_name}")
            else:
                ax_time.plot(x_data, y_data, marker=markers[sol_name],
                             linestyle=linestyles[sol_name], color=colors[sol_name],
                             linewidth=1, label=f"{sol_name}")

        # Set labels for y-axis
        ax_time.set_ylabel(f'{coord} (mm)')
        ax_time.grid(True, alpha=0.3)

        # Add horizontal reference line at zero
        ax_time.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Set y-limits with padding
        padding = 0.1 * (y_maxs[i] - y_mins[i])
        if i == 2:  # Z component might have different scale
            y_min_padded = -10
            y_max_padded = 10
        else:  # X and Y components
            y_min_padded = -6
            y_max_padded = 6

        # Override with actual data range if it's larger
        if y_mins[i] < y_min_padded:
            y_min_padded = y_mins[i] - padding
        if y_maxs[i] > y_max_padded:
            y_max_padded = y_maxs[i] + padding

        ax_time.set_ylim(y_min_padded, y_max_padded)

        # Format x-axis date labels
        ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_time.xaxis.set_major_locator(mdates.YearLocator())

        # Only add x-label to bottom subplot
        if i == n_plots - 1:
            ax_time.set_xlabel('Date')
        plt.setp(ax_time.xaxis.get_majorticklabels(), rotation=45)

        # Add legend to first subplot only
        if i == 0:
            ax_time.legend(loc=legend_loc, framealpha=0.7)

        # Create Lomb-Scargle periodogram axes (1 column wide)
        ax_ls = fig.add_subplot(gs[i, 2])
        ls_axes.append(ax_ls)

        # Set title and labels for Lomb-Scargle plot
        ax_ls.set_title(f'Periodogram: {coord} Motion')
        ax_ls.set_xlabel('Period (days)')
        ax_ls.set_ylabel('Amplitude (mm)')

        # Logarithmic x-axis scale
        ax_ls.set_xscale('log')
        ax_ls.set_xlim(min_period, max_period)

        # Set y-limit with padding for amplitude
        amp_padding = 0.1 * amp_maxs[i]
        ax_ls.set_ylim(0, amp_maxs[i] + amp_padding)

        # Calculate and plot periodogram for each solution
        for sol_name, df in solutions_dict.items():
            if coord not in df.columns:
                continue

            # Skip if less than 5 data points
            if len(df) < 5:
                continue

            # Get time series data and remove NaNs
            y_data = df[coord].dropna()
            if len(y_data) < 5:
                continue

            x_dates = y_data.index

            # Convert dates to days since the first point
            x_days = np.array([(d - x_dates[0]).total_seconds() / (24 * 3600) for d in x_dates])
            y_vals = y_data.values

            # Calculate Lomb-Scargle periodogram
            # Define periods and frequencies
            periods = np.logspace(np.log10(min_period), np.log10(max_period), n_freqs)
            freqs = 1 / periods

            try:
                # Calculate periodogram for unevenly sampled data
                pgram = lombscargle(x_days, y_vals - np.mean(y_vals), 2 * np.pi * freqs, normalize=True)

                # Convert to amplitude (assuming data is already in mm)
                amplitudes = np.sqrt(4 * pgram / len(y_vals))

                # Plot periodogram
                ax_ls.plot(periods, amplitudes, color=colors[sol_name],
                           linestyle=linestyles[sol_name], alpha=0.8,
                           label=f"{sol_name}")
            except Exception as e:
                print(f"Error computing periodogram for {sol_name}, {coord}: {e}")

        # Add grid to periodogram
        ax_ls.grid(True, which='both', alpha=0.3)

        # Add vertical lines at common periods of interest
        common_periods = [365.25, 182.63, 121.75, 91.31]  # Annual, semiannual, seasonal, etc.
        period_labels = ['Annual', 'Semi-annual', '4-month', '3-month']

        for period, label in zip(common_periods, period_labels):
            if min_period <= period <= max_period:
                ax_ls.axvline(x=period, color='gray', linestyle='--', alpha=0.5)
                ax_ls.text(period, ax_ls.get_ylim()[1] * 0.95, f"{label}\n({period:.0f}d)",
                           ha='right', va='top', rotation=90, fontsize=8, alpha=0.7)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Add overall title
    plt.suptitle('Geocenter Motion and Periodogram Analysis',
                 fontsize=16, y=0.995)

    # Adjust for suptitle
    plt.subplots_adjust(top=0.93)

    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    return fig, time_axes, ls_axes

def get_geocenter_motion(solution, degree, date_start, date_end, base_dir):

    degree1_z = create_coefficient_time_series(base_dir,
                                             solution=solution, degmax=degree,
                                             coeff_type='load', degree=1, order=0,
                                             start_date=date_start,end_date=date_end)
    degree1_xy = create_coefficient_time_series(base_dir,
                                             solution=solution, degmax=degree,
                                             coeff_type='load', degree=1, order=1,
                                             start_date=date_start,end_date=date_end)

    degree1_z.rename({'C':'C10','C_error':'C10_error'},axis=1,inplace=True)
    degree1_z.drop(['S','S_error'],axis=1,inplace=True)
    degree1_xy.rename({'C':'C11','C_error':'C11_error',
                              'S':'S11','S_error':'S11_error'},axis=1,inplace=True)

    degree1 = degree1_z.merge(degree1_xy,on='date')
    degree1.drop(['date_str_x','date_str_y'],axis=1,inplace=True)

    RHO_E = 5517.0  # Average density of Earth in kg/m³
    K1_CF = 0.021  # Load Love number for degree 1
    factor = - (1 + K1_CF) / RHO_E *1e3

    degree1[['X','Y','Z']] =  degree1[['C11','S11','C10']]* factor
    degree1[['X_error', 'Y_error', 'Z_error']] = degree1[['C11_error', 'S11_error', 'C10_error']] * factor
    return degree1

from scipy import stats
from astropy.timeseries import LombScargle
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.ticker import AutoMinorLocator


def plot_series_with_lombscargle(data_dict, column_name, apply_lowpass_filter, title=None, units=None,
                                 y_offset=0, periodogram_offset=0, figsize=(12, 8), width_cm=None,
                                 colors=None, min_period=2, max_period=500, alpha_raw=0.7,
                                 alpha_filtered=0.8, line_width_raw=1, line_width_filtered=1.2,
                                 cutoff_period=150, save_path=None, return_fig=False):
    """
    Plot multiple time series with their Lomb-Scargle periodograms and return statistical information.

    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are series names and values are pandas DataFrames with datetime index
    column_name : str
        Name of the column to analyze in each DataFrame
    apply_lowpass_filter : function
        Function to apply low-pass filter, with signature apply_lowpass_filter(signal, cutoff_freq, sampling_rate)
    title : str, optional
        Title for the plot
    units : dict or str, optional
        Units for the series. If dict, keys should match data_dict keys. If str, same unit for all series
    y_offset : float or dict, optional
        Vertical offset for each time series plot. If dict, keys should match data_dict keys
    periodogram_offset : float or dict, optional
        Vertical offset for each periodogram plot. If dict, keys should match data_dict keys
    figsize : tuple, optional
        Figure size (width, height) in inches
    width_cm : float, optional
        Width of the figure in centimeters (overrides figsize width if provided)
    colors : dict or list, optional
        Colors for the time series and periodograms. If dict, keys should match data_dict keys
    min_period : float, optional
        Minimum period (in days) for the periodogram
    max_period : float, optional
        Maximum period (in days) for the periodogram
    alpha_raw : float, optional
        Alpha (transparency) value for the raw time series
    alpha_filtered : float, optional
        Alpha value for the filtered time series
    line_width_raw : float, optional
        Line width for the raw time series
    line_width_filtered : float, optional
        Line width for the filtered time series
    cutoff_period : float, optional
        Low-pass filter cutoff period in days
    save_path : str, optional
        Path to save the figure (e.g., 'my_plot.png'). If None, figure is not saved.
    return_fig : bool, optional
        If True, returns the figure object

    Returns:
    --------
    dict
        Dictionary containing statistical information about each time series
    matplotlib.figure.Figure, optional
        Figure object if return_fig is True
    """
    # Validate inputs and set defaults
    if not isinstance(data_dict, dict):
        raise ValueError("data_dict must be a dictionary of DataFrames")

    n_series = len(data_dict)

    # Convert width_cm to inches if provided
    if width_cm is not None:
        # Convert cm to inches (1 inch = 2.54 cm)
        width_inches = width_cm / 2.54
        # Maintain aspect ratio from figsize
        height_inches = width_inches * (figsize[1] / figsize[0])
        figsize = (width_inches, height_inches)

    # Set default colors if not provided
    if colors is None:
        default_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = {series_key: default_colors[i % len(default_colors)] for i, series_key in enumerate(data_dict.keys())}
    elif not isinstance(colors, dict):
        # If colors is a list, convert to dict
        if isinstance(colors, (list, tuple)):
            colors = {series_key: colors[i % len(colors)] for i, series_key in enumerate(data_dict.keys())}
        else:
            # If colors is a single color, use for all series
            colors = {series_key: colors for series_key in data_dict.keys()}

    # Handle y_offset
    if not isinstance(y_offset, dict):
        y_offset = {series_key: i * y_offset if y_offset != 0 else 0
                    for i, series_key in enumerate(data_dict.keys())}

    # Handle periodogram_offset
    if not isinstance(periodogram_offset, dict):
        periodogram_offset = {series_key: i * periodogram_offset if periodogram_offset != 0 else 0
                              for i, series_key in enumerate(data_dict.keys())}

    # Handle units
    if units is None:
        units = {series_key: "" for series_key in data_dict.keys()}
    elif not isinstance(units, dict):
        units = {series_key: units for series_key in data_dict.keys()}

    # Create a figure with two subplots
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1])

    # Time series plot
    ax1 = fig.add_subplot(gs[0])

    # Lomb-Scargle periodogram plot
    ax2 = fig.add_subplot(gs[1])

    # Statistics dictionary to return
    stats_dict = {}

    # Process each series
    for i, (series_key, data) in enumerate(data_dict.items()):
        # Convert to DataFrame if it's a Series
        if isinstance(data, pd.Series):
            data = data.to_frame(name=column_name)

        # Check if the data has a datetime index, if not try to convert
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except:
                raise ValueError(f"Series {series_key} must have a datetime index or be convertible to datetime")

        # Extract the time series - use the specified column name
        if column_name in data.columns:
            series = data[column_name]
        else:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame for '{series_key}'")

        # Calculate the time in days since the first observation for Lomb-Scargle
        time_days = np.array([(d - data.index[0]).total_seconds() / (24 * 3600) for d in data.index])

        # Get the current color
        color = colors[series_key]

        # Get current offsets
        curr_y_offset = y_offset[series_key]
        curr_p_offset = periodogram_offset[series_key]

        # Plot the raw time series
        ax1.plot(data.index, series + curr_y_offset, color=color, alpha=alpha_raw,
                 linewidth=line_width_raw, label='__nolegend__')

        # Apply the low-pass filter using the provided function
        cutoff_frequency = 1 / cutoff_period  # Convert to frequency in cycles/day
        sampling_rate = 1  # Assume 1 sample per day (we'll resample if needed)

        # Check if the data is evenly sampled
        is_evenly_sampled = np.allclose(np.diff(time_days), np.diff(time_days).mean(), rtol=0.05)

        if is_evenly_sampled:
            # For evenly sampled data, directly apply the filter
            filtered_signal = apply_lowpass_filter(series.values, cutoff_frequency, sampling_rate)
        else:
            # For unevenly sampled data, interpolate to a uniform grid, filter, then interpolate back
            # Create a uniform time grid
            uniform_time = np.linspace(time_days.min(), time_days.max(), len(time_days))

            # Interpolate to uniform grid
            from scipy.interpolate import interp1d
            f = interp1d(time_days, series.values, kind='linear', bounds_error=False, fill_value="extrapolate")
            uniform_signal = f(uniform_time)

            # Apply filter to uniform signal
            filtered_uniform = apply_lowpass_filter(uniform_signal, cutoff_frequency, sampling_rate)

            # Interpolate back to original time points
            f_back = interp1d(uniform_time, filtered_uniform, kind='linear', bounds_error=False,
                              fill_value="extrapolate")
            filtered_signal = f_back(time_days)

        # Plot the filtered time series
        ax1.plot(data.index, filtered_signal + curr_y_offset, color=color, alpha=alpha_filtered,
                 linewidth=line_width_filtered, label='__nolegend__')

        # Calculate periods from min_period to max_period days for the periodogram plot
        periods = np.logspace(np.log10(min_period), np.log10(max_period), 1000)
        frequencies_ls = 1 / periods  # Convert to frequencies (cycles/day)

        # Calculate the Lomb-Scargle periodogram
        ls = LombScargle(time_days, series.values)
        power = ls.power(frequencies_ls)

        # Convert to amplitude
        amplitude = np.sqrt(power) * np.std(series.values) * 2

        # Plot the periodogram with period on x-axis
        ax2.plot(periods, amplitude + curr_p_offset, color=color, label=series_key)

        # Statistical analysis for this series
        stats_dict[series_key] = {
            'mean': series.mean(),
            'median': series.median(),
            'std_dev': series.std(),
            'min': series.min(),
            'max': series.max(),
            'range': series.max() - series.min(),
            'skewness': stats.skew(series.dropna()),
            'kurtosis': stats.kurtosis(series.dropna()),
            'n_observations': len(series),
            'missing_values': series.isna().sum()
        }

        # Find the top periods based on amplitude
        top_indices = np.argsort(amplitude)[-5:]  # Get indices of 5 highest amplitudes
        top_periods = periods[top_indices]
        top_amplitudes = amplitude[top_indices]

        # Sort by period (largest to smallest)
        sort_idx = np.argsort(top_periods)[::-1]
        top_periods = top_periods[sort_idx]
        top_amplitudes = top_amplitudes[sort_idx]

        stats_dict[series_key]['top_periods_days'] = top_periods
        stats_dict[series_key]['top_periods_amplitudes'] = top_amplitudes
        stats_dict[series_key]['top_periods_years'] = top_periods / 365.25

    # Add a vertical line at the cutoff period in the periodogram (no label)
    ax2.axvline(x=cutoff_period, color='black', linestyle=':', alpha=0.5)

    # Set labels and titles
    unit_text = next(iter(units.values())) if any(units.values()) else ""
    ax1.set_ylabel(f'{title}{" [" + unit_text + "]" if unit_text else ""}')

    ax1.grid(True, which='major', alpha=0.3, axis='y')
    ax1.legend(loc='upper right',fontsize=7)

    # Configure date axis formatting
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # ax1.xaxis.set_major_locator(mdates.YearLocator())
    #
    # # Add minor ticks for months
    # ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    #
    # # Make sure x-axis tick labels are visible
    # plt.setp(ax1.get_xticklabels(), visible=True)
    #
    # # Ensure tick labels don't overlap
    # fig.autofmt_xdate(rotation=45, ha='right')

    # Periodogram plot settings
    ax2.set_xscale('log')
    ax2.set_xlabel('Period (days)')
    ax2.set_ylabel(f'Amplitude{" [" + unit_text + "]" if unit_text else ""}')
    # ax2.set_title(f'Lomb-Scargle Periodogram (Low-pass filter: {cutoff_period} days)')
    ax2.legend(loc='upper left')

    # Add grid with specific period markers
    ax2.grid(True, which='major', alpha=0.3, axis='y')

    # Add vertical lines at specific periods of interest (annual, semi-annual, etc.) without labels
    periods_of_interest = [365.25, 182.625, 121.75, 91.3125, 30.4375, 14, 7]

    for period in periods_of_interest:
        if min_period <= period <= max_period:
            ax2.axvline(x=period, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if return_fig:
        return stats_dict, fig
    else:
        return stats_dict


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def plot_phasors(df, component, figsize=(10, 8), width_cm=None, save_path=None, return_fig=False):
    """
    Plot phasors based on 1cpy_amp and 1cpy_phase with errors for the specified component.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the dataset with columns for dataset name, component,
        1cpy_amp, 1cpy_amp_e, 1cpy_phase, 1cpy_phase_e
    component : str
        Component to filter the data on (e.g., 'C10', 'C11', 'S11')
    figsize : tuple, optional
        Figure size (width, height) in inches
    width_cm : float, optional
        Width of the figure in centimeters (overrides figsize width if provided)
    save_path : str, optional
        Path to save the figure (e.g., 'phasor_plot.png'). If None, figure is not saved.
    return_fig : bool, optional
        If True, returns the figure object

    Returns:
    --------
    matplotlib.figure.Figure, optional
        Figure object if return_fig is True
    """
    # Filter data for the specific component
    filtered_df = df[df['Component'] == component].copy()

    if filtered_df.empty:
        raise ValueError(f"No data found for component '{component}'")

    # Convert width_cm to inches if provided
    if width_cm is not None:
        # Convert cm to inches (1 inch = 2.54 cm)
        width_inches = width_cm / 2.54
        # Maintain aspect ratio from figsize
        height_inches = width_inches * (figsize[1] / figsize[0])
        figsize = (width_inches, height_inches)

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Convert phase from degrees to radians
    filtered_df['1cpy_phase_rad'] = np.radians(filtered_df['1cpy_phase'])
    filtered_df['1cpy_phase_e_rad'] = np.radians(filtered_df['1cpy_phase_e'])

    # Maximum amplitude for plot scaling - handle very small values
    max_amp = filtered_df['1cpy_amp'].max()

    # Format the amplitude values based on their magnitude
    if max_amp < 1e-6:
        # For very small values, use scientific notation
        format_amp = lambda x: f'{x:.1e}'
        # Round up to the next multiple of 10^n where n is the order of magnitude
        order = np.floor(np.log10(max_amp))
        max_amp = np.ceil(max_amp / (10 ** order)) * (10 ** order)
    else:
        format_amp = lambda x: f'{x:.1f}'
        max_amp = np.ceil(max_amp)

    # Calculate x and y coordinates for phasors
    filtered_df['x'] = filtered_df['1cpy_amp'] * np.cos(filtered_df['1cpy_phase_rad'])
    filtered_df['y'] = filtered_df['1cpy_amp'] * np.sin(filtered_df['1cpy_phase_rad'])

    # Add circular grid - ensure we have 4 circles even for very small values
    radii = np.linspace(max_amp / 4, max_amp, 4)
    for r in radii:
        circle = plt.Circle((0, 0), r, fill=False, color='gray', linestyle='--', alpha=0.3)
        ax.add_artist(circle)
        # Add radius label with appropriate formatting
        ax.text(r * np.cos(np.pi / 8), r * np.sin(np.pi / 8), format_amp(r),
                ha='left', va='bottom', alpha=0.7, fontsize=8)

    # Plot individual phasors
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_df)))

    for idx, (_, row), color in zip(range(len(filtered_df)), filtered_df.iterrows(), colors):
        # Plot phasor as point (smaller size)
        ax.scatter(row['x'], row['y'], s=20, color=color, marker='x', zorder=3, label=row['Dataset'])

    # Set up the axes - REMOVED BLACK AXIS LINES
    ax.set_xlim(-max_amp * 1.2, max_amp * 1.2)  # Extra space for labels
    ax.set_ylim(-max_amp * 1.2, max_amp * 1.2)

    # Remove all spines (including the center ones)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Remove tick marks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add phase angle lines and annotations
    for angle_deg in range(0, 360, 45):
        angle_rad = np.radians(angle_deg)
        # Add radial line
        ax.plot([0, max_amp * np.cos(angle_rad)], [0, max_amp * np.sin(angle_rad)],
                'k-', alpha=0.1)
        # Add angle label
        x = 1.1 * max_amp * np.cos(angle_rad)
        y = 1.1 * max_amp * np.sin(angle_rad)
        ax.text(x, y, f'{angle_deg}°', ha='center', va='center', alpha=0.7, fontsize=8)

    # Set title with amplitude scale indication for very small values
    if max_amp < 1e-6:
        title = f"{component} (values in scientific notation)"
    else:
        title = f"{component}"
    ax.set_title(title, pad=20)

    # Calculate number of columns for legend based on number of datasets
    n_datasets = len(filtered_df)
    ncol = min(3, n_datasets)  # Maximum 3 columns, or fewer if less datasets

    # Add legend above the plot
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                       ncol=ncol, frameon=True,
                       fancybox=True, shadow=True, fontsize=8)

    # Equal aspect ratio
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if return_fig:
        return fig
    else:
        plt.show()
        return None  # Fixed the return behavior