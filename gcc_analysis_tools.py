import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import spectrogram
import os
from scipy.stats import median_abs_deviation as mad
from scipy import linalg
from scipy.stats import zscore
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.signal import butter, filtfilt
import seaborn as sns
import matplotlib.patheffects as path_effects
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import powerlaw
from scipy import signal
import matplotlib.patches as mpatches
import geodezyx.conv.conv_time as ct
from datetime import datetime

plt.interactive(False)

plt.rcParams.update({
        'font.size': 9,        # Set all font sizes to 10
        'font.family': 'Arial', # Set font to Arial
        'figure.dpi': 300
    })

# Define colors for each file
def name_to_color(name):
    colors = {
        'jpl': 'red',
        'mit': 'blue',
        'ngs': 'green',
        'cod': 'orange',
        'esa': 'purple',
        'gfz': 'brown',
        'grg': 'pink',
        'ig3': 'darkturquoise',
        'tug': 'm',
        'whu': 'lightgreen',
        'ulr': 'lightblue',
        'vce': 'g',
        'slr': 'indianred',
        'tn13': 'orangered',
        'cheng': 'm',
        'cheng_lp': 'b',
        'vceM6': 'g',
        'vceM1': 'm',
        'vceM3': 'b',
        'vceM6': 'g',
        '1DW': 'g',
        '1DWA': 'orange',
        '1DLOAD_ALL': 'r',
        '1DLOAD_DAT': 'indianred',
        '1DE': 'm',
        '1DEA': 'b',
        '7DEQ': 'b',
        '7DW': 'purple',
        '7DWA': 'plum',
        '7DEA': 'maroon',
        'vceY1': 'purple',
        'Yu':'teal',
        'ITRF2020':'plum',
        'UPWr':'maroon',
        'F20': 'red',
        'F20': 'orange',
        'FIX90': 'blue',
        'EMP': 'purple',
        'CORE': 'm',
        'KMEANS': 'c'

    }
    return colors.get(name,'k')

def name_to_color(name):
    colors = {
        'ig3': 'darkturquoise',
        'slr': 'indianred',
        'tn13': 'orangered',
        'AHOS': 'tab:brown',
        'A': 'tab:orange',
        'O': 'tab:blue',
        'H': 'tab:green',
        'S': 'tab:purple',
        'cheng': 'm',
        'cheng_lp': 'b',
        'S-D-W-X-1': 'g',
        'S-D-W-X-7': 'g',
        'S-A-W-X-1': 'orange',
        'S-A-W-X-7': 'orange',
        'L-D-W-H-1': 'b',
        'L-D-W-H-7': 'b',
        'L-A-W-H-1': 'm',
        'L-A-W-H-7': 'm',
        'L-D-W-AHOS-1': 'r',
        'L-D-W-AHOS-7': 'r',
        'L-A-W-AHOS-1': 'purple',
        'L-A-W-AHOS-7': 'purple',
    }
    return colors.get(name,'k')

def reverse_cf_cm(df):
    df.loc[:,['X','Y','Z']] *= -1
    return df

def apply_lowpass_filter(signal, cutoff_frequency, sampling_rate, order=4):
    """
    Apply a Butterworth low-pass filter to the time-series data.

    Parameters:
    signal (array-like): The time-series data to smooth.
    cutoff_frequency (float): The cutoff frequency for the filter (in Hz).
    sampling_rate (float): The sampling rate of the data (in Hz).
    order (int): The order of the filter (default is 4).

    Returns:
    array-like: The smoothed time-series data after applying the low-pass filter.
    """
    nyquist = 0.5 * sampling_rate  # Nyquist frequency
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Get filter coefficients
    filtered_signal = filtfilt(b, a, signal)  # Apply forward-backward filtering for zero-phase filtering
    return filtered_signal

def resample_and_interpolate(dataframes_dict,sampling='M'):
    """
    This function takes a dictionary of DataFrames, resamples each DataFrame to daily frequency,
    and interpolates any missing values using the mean of the surrounding data.
    It also prints the number of missing values before interpolation.

    Args:
        dataframes_dict (dict): Dictionary of DataFrames with DatetimeIndex.

    Returns:
        dict: Dictionary with resampled and interpolated DataFrames.
    """
    resampled_dict = {}

    for key, df in dataframes_dict.items():
        # Ensure the index is a DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df[df.select_dtypes(include=['number']).columns]

        # Resample to daily frequency, filling in missing dates
        df_resampled = df.resample(sampling).mean()

        # Count the number of missing values before interpolation
        missing_count = df_resampled.isnull().sum().sum()

        # Interpolate missing values using the mean of the neighboring values
        df_interpolated = df_resampled.interpolate(method='linear')


        # Print the number of missing values found and filled
        print(f"{key}: {missing_count} missing values were filled.")

        # df[['gpsweek', 'week_dow']] = df.index.map(lambda dt: pd.Series(ct.dt2gpstime(dt)))
        # df['MJD'] = df.index.map(ct.dt2MJD)

        # Store the processed DataFrame in the new dictionary
        resampled_dict[key] = df_interpolated

    return resampled_dict

def merge_and_unstack(data_dict, value_vars = ['X', 'Y', 'Z']):
    # List to hold the reformatted DataFrames
    reformatted_dfs = []

    # Loop over the dictionary and reformat each DataFrame
    for key, df in data_dict.items():
        dfindex = df.reset_index()
        # Melt the X, Y, Z columns into a single column 'value' and create a 'type' column
        melted_df = dfindex.melt(id_vars=['ISO_date','gpsweek', 'MJD'],
                            value_vars=value_vars,
                            var_name='type',
                            value_name='value')
        # Add the type of data source (key) as a column
        melted_df['source'] = key

        # Append the transformed DataFrame to the list
        reformatted_dfs.append(melted_df)

    # Merge all DataFrames in the list
    merged_df = pd.concat(reformatted_dfs)

    return merged_df


def resample_daily_midnight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a DataFrame with a datetime index to have only daily values at midnight,
    using linear interpolation.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a datetime index.

    Returns:
    pd.DataFrame: Resampled DataFrame with only daily data at midnight.
    """
    # Ensure the index is a datetime index
    df = df.sort_index()
    df = df.asfreq('H')
    df = df.interpolate(method='linear')  # Linear interpolation for missing values

    # Resample to midnight only
    df_resampled = df.resample('D').first()

    return df_resampled

def vce(y_raw, vce_method = 'forstner', iterate = True ,  tol = 1e-5, maxiter = 1000):
    """
    VCE function

    Returns
    -------
    s2 : array
        Estimated variance factors
    rms : array
        RMS of "AC - combined" orbit differences

    Parameters
    ----------
    y : array
        (n x m) array of AC orbits. (n=96*3=288, m=number of ACs)

    """

    # Initializations
    (n, m) = y_raw.shape
    s20 = np.ones(m)
    s2 = s20.copy()
    ds2 = np.zeros(m)
    rms = np.zeros(m)
    niter = 0

    y = remove_outliers(y_raw).values
    # f, a = plt.subplots(3, 1)
    # for comp, dfcomp, comp_cleaned, dfcomp_cleaned, ax in zip(y_raw.groupby('type'), y.groupby('type'), a):
    #     ax.plot(dfcomp)

    data_len = dict(removed = len(y_raw)-len(y),
                    lenvce = len(y))
    # Initial weighted mean
    x = np.sum(y / s2, axis=1) / np.sum(1 / s2)

    # Start iterations
    end = False
    while not (end):
        niter += 1

        # Update variance factors
        for i in range(m):
            if vce_method == 'mad':
                ds2[i] = mad(y[:, i] - x, scale='normal') ** 2 / s2[i]
            elif vce_method == 'forstner':
                ds2[i] = np.sum((y[:, i] - x) ** 2) / n / (1 - 1 / s2[i] / np.sum(1 / s2)) / s2[i]

            s2[i] *= ds2[i]

        # Update weighted mean orbit
        x = np.sum(y / s2, axis=1) / np.sum(1 / s2)

        # RMS of "AC - combined" orbit differences
        for i in range(m):
            rms[i] = np.sqrt(np.sum((y[:, i] - x) ** 2) / n)

        # Stop iterations if needed
        if not (iterate) or (np.max(np.abs(np.log(ds2))) < tol) or (niter > maxiter):
            # print(np.max(np.abs(np.log(ds2))), niter,rms)
            end = True

    return (s2, rms, x, data_len)


def filter_single_dataframe(df, start_date, end_date, column=None):
    """
    Filters a single DataFrame between the given start and end date.

    :param df: DataFrame with a DateTimeIndex
    :param start_date: Start date (string in 'yyyy-mm-dd' format) for filtering
    :param end_date: End date (string in 'yyyy-mm-dd' format) for filtering
    :return: Filtered DataFrame
    """
    # Convert start_date and end_date to pandas datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Check if the DataFrame has a datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        # Ensure the DataFrame index is timezone-aware (convert if naive)
        return df[(df.index >= start_date) & (df.index <= end_date)]
    elif column:
        return df[(df[column] >= start_date) & (df[column] <= end_date)]

    return df  # Return unchanged if it doesn't have a DateTimeIndex

def filter_dataframes_dict(dataframes_dict, start_date, end_date, column=None):
    """
    Filters a dictionary of DataFrames between the given start and end date.

    :param dataframes_dict: Dictionary of DataFrames
    :param start_date: Start date (string in 'yyyy-mm-dd' format) for filtering
    :param end_date: End date (string in 'yyyy-mm-dd' format) for filtering
    :return: Dictionary of filtered DataFrames
    """
    filtered_dataframes = {}

    for key, df in dataframes_dict.items():
        filtered_dataframes[key] = filter_single_dataframe(df, start_date, end_date, column)

    return filtered_dataframes


def remove_outliers(data: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
    """
    Identify and remove rows containing outliers in the dataset based on Z-scores.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical features.
    threshold (float): The Z-score threshold to identify outliers. Default is 3.

    Returns:
    pd.DataFrame: A DataFrame with the outliers removed.
    """
    # Calculate Z-scores for each feature
    data = data.subtract(data.median(axis=1), axis=0)
    z_scores = data.apply(zscore)

    # Identify rows where any feature exceeds the threshold for outliers
    outliers = (z_scores > threshold) | (z_scores < -threshold)

    # Keep only rows that don't have any outliers
    cleaned_data = data[~outliers.any(axis=1)].reset_index(drop=True)
    return cleaned_data

# Helper function to load data with specified columns
def load_data(file_path):
    columns = ['gpsweek', "week_dow", "MJD", "ISO_date", "X", "Y", "Z", "sigmaX", "sigmaY", "sigmaZ"]
    df = pd.read_csv(file_path, delim_whitespace=True, names=columns, parse_dates=['ISO_date'])
    df = df.set_index('ISO_date')
    df[["X", "Y", "Z"]] *= -1
    return df

# Function to fit annual and semiannual signals
def fit_function(t, a0, a1, b1, a2, b2):
    return a0 + a1 * np.cos(2 * np.pi * t / 365.25) + b1 * np.sin(2 * np.pi * t / 365.25) + \
        a2 * np.cos(4 * np.pi * t / 365.25) + b2 * np.sin(4 * np.pi * t / 365.25)

def annual_semiannual_model(t, a0, a1, b1, a2, b2):
    return a0 + a1 * np.cos(2 * np.pi * t) + b1 * np.sin(2 * np.pi * t) + \
                 a2 * np.cos(4 * np.pi * t) + b2 * np.sin(4 * np.pi * t)
# Function for sliding window FFT analysis
def sliding_window_fft(df, window_size=365, step_size=30, component='X'):
    n = len(df)
    time_step = 1  # Assuming daily sampled data
    periods = []
    amplitudes = []
    phases = []

    for start in range(0, n - window_size + 1, step_size):
        window = df[component].iloc[start:start + window_size].to_numpy()
        window = window - np.nanmean(window)
        window = np.nan_to_num(window)

        fft_vals = rfft(window)
        freqs = rfftfreq(window_size, time_step)

        # Find the frequency closest to the annual signal (1 cycle per year)
        idx = np.argmin(np.abs(freqs - 1 / 365.25))
        amplitude = 2* np.abs(fft_vals[idx]) / window_size
        phase = np.angle(fft_vals[idx])

        periods.append(df.index[start + window_size // 2])
        amplitudes.append(amplitude)
        phases.append(phase)

    return periods, amplitudes, phases

# Function for harmonic analysis over time
def harmonic_analysis_over_time(df, window_size=365, step_size=30, component='X'):
    n = len(df)
    time = (df.index - df.index[0]).days
    amplitudes = []
    phases = []
    periods = []

    for start in range(0, n - window_size + 1, step_size):
        window_time = time[start:start + window_size]
        window = df[component].iloc[start:start + window_size]
        valid_idx = ~window.isna()

        popt, _ = curve_fit(fit_function, window_time[valid_idx], window[valid_idx])
        a1, b1 = popt[1], popt[2]
        amplitude = np.sqrt(a1 ** 2 + b1 ** 2)
        phase = np.arctan2(b1, a1)

        amplitudes.append(amplitude)
        phases.append(phase)
        periods.append(df.index[start + window_size // 2])

    return periods, amplitudes, phases


# Function for time-frequency spectrogram
def time_frequency_spectrogram(df, component='X', window='hann', nperseg=365, noverlap=182):
    series = df[component].to_numpy()
    series = series - np.nanmean(series)
    series = np.nan_to_num(series)

    freqs, times, Sxx = spectrogram(series, fs=1, window=window, nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    return freqs, times, Sxx

def to_common_dates(dataframes):
    # Convert all dates to timezone-naive (remove timezone information)
    min_dates = [df.index.min().tz_localize(None) for df in dataframes.values()]
    max_dates = [df.index.max().tz_localize(None) for df in dataframes.values()]
    # Calculate the common time range
    common_start = max(min_dates)
    common_end = min(max_dates)
    print(f'NEW DATE RANGE {common_start} - {common_end}')
    for name, df in dataframes.items():
        df.index = df.index.tz_localize(None)
        df_common = df[(df.index >= common_start) & (df.index <= common_end)]
        dataframes[name] = df_common
    return dataframes


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

def plot_time_series_with_fits(dataframes, offsets={'X': 20, 'Y': 20, 'Z': 50}, plotname='TS_PLOT',
                               ylim=None):
    # Set font properties

    # Set the figure size to 8.4 cm x 12 cm and dpi to 300
    fig, axs = plt.subplots(3, 1, figsize=(15/2.54, 18/2.54), sharex=True, dpi=300)  # Convert cm to inches by dividing by 2.54

    # Plot each series
    for name, df in dataframes.items():
        # Filter data to the common time range
        df_common = df.copy()

        time = (df_common.index - df_common.index[0]).days  # Time in days since the start
        for i, component in enumerate(['X', 'Y', 'Z']):
            offset = offsets[component] * list(dataframes.keys()).index(name)
            color = name_to_color(name)  # Get color from name_to_color function

            # Plot the data with reduced linewidth and transparency
            axs[i].plot(df_common.index, df_common[component] + offset,
                        label=name, color=color, linewidth=0.8, alpha=.7)

            # Fit annual and semiannual signals
            valid_idx = ~df_common[component].isna()
            popt, _ = curve_fit(fit_function, time[valid_idx], df_common[component][valid_idx])
            fitted_signal = fit_function(time, *popt)

            # Plot the fitted signal with a thicker dashed line
            axs[i].plot(df_common.index, fitted_signal + offset, label=f'__NoLegend', color='k', linestyle='-', linewidth=2,alpha=0)

            #lowpass
            cutoff_period = 150  # days
            cutoff_frequency = 1 / cutoff_period  # Convert to frequency in Hz
            sampling_rate = 1  # Data is sampled every 1 day, so the sampling rate is 1 Hz
            lowpass_filtered_signal = apply_lowpass_filter(df_common[component].values, cutoff_frequency, sampling_rate)

            # Plot the low-pass filtered signal with a solid thick line
            axs[i].plot(df_common.index, lowpass_filtered_signal + offset,
                        label=f'__NoLegend', color='k', linestyle='-', linewidth=1.2, alpha=0.8)

    # Set the legend above the first subplot only
    leg = axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=10,
                  handletextpad=0.3, columnspacing=0.1, handlelength=0.3, handleheight=1.5)  # Adjust spacing and handle length/height
    for legobj in leg.get_lines():
        legobj.set_linewidth(2.0)

    # Set the y-ticks (major ticks every offset and minor ticks every half offset)
    for i, component in enumerate(['X', 'Y', 'Z']):
        offset = offsets[component]

        # Set major y-ticks to be every `offset`
        axs[i].yaxis.set_major_locator(MultipleLocator(offset))

        # Set minor y-ticks to be every `offset / 2`
        axs[i].yaxis.set_minor_locator(MultipleLocator(offset / 2))

        # Add grid lines for both major and minor ticks
        axs[i].grid(True, which='major', linewidth=1)
        axs[i].grid(True, which='minor', linestyle='--', linewidth=0.5)

        axs[i].autoscale(enable=True, axis='x', tight=True)

        if ylim:
            axs[i].set_ylim(ylim[component][0],ylim[component][1])

    # Add labels to the y-axis
    axs[0].set_ylabel('X (mm)')
    axs[1].set_ylabel('Y (mm)')
    axs[2].set_ylabel('Z (mm)')

    # Add label to the x-axis on the last subplot
    axs[2].set_xlabel('Date')

    # Reduce space between subplots
    plt.subplots_adjust(wspace=0.1,hspace=0.1)  # Decrease vertical space between subplots

    plt.tight_layout(pad=.1)
    # plt.show()
    save_dir = './PLOTS/TS/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir,plotname+'.png'))
    plt.show()
    return fig, axs

def fit_annual_in_windows(data,window,overlap):
    amplitudes = {}
    phases = {}
    for label, data_series in data.items():
        # Create a list to store the results of the 3-year windows
        years = pd.date_range(start=data_series.index.min(), end=data_series.index.max(), freq='Y').year
        # Calculate the window ranges and overlap
        window_amplitudes = {}
        window_phases = {}
        for start_year in range(years.min(), years.max() - window + 2, overlap):
            end_year = start_year + window - 1
            window_data = data_series[(data_series.index.year >= start_year) & (data_series.index.year <= end_year)]
            time = (window_data.index - window_data.index[0]).days
            amps = {}
            phs = {}
            for component in ['X','Y','Z']:
                window_data_comp = window_data[component]
                popt, _ = curve_fit(fit_function, time, window_data_comp)
                a1, b1 = popt[1], popt[2]
                amplitude = np.sqrt(a1 ** 2 + b1 ** 2)
                phase = np.degrees(np.arctan2(b1, a1))
                amps[component] = amplitude
                phs[component] = phase
            # Append results
            window_amplitudes[f'{start_year}-{end_year}'] = amps
            window_phases[f'{start_year}-{end_year}'] = phs
        amplitudes[label] = window_amplitudes
        phases[label] = window_phases
    return amplitudes, phases

def fit_noise_in_windows(data,window,overlap):
    bics = {}
    for label, data_series in data.items():
        # Create a list to store the results of the 3-year windows
        years = pd.date_range(start=data_series.index.min(), end=data_series.index.max(), freq='Y').year
        # Calculate the window ranges and overlap
        window_bics = {}
        for start_year in range(years.min(), years.max() - window + 2, overlap):
            end_year = start_year + window - 1
            window_data = data_series[(data_series.index.year >= start_year) & (data_series.index.year <= end_year)]
            window_bic = {}
            for component in ['X','Y','Z']:
                window_data_comp = window_data[component]
                window_bic[component] = best_fit_noise_model(window_data_comp, criteria='bic')
            # Append results
            window_bics[f'{start_year}-{end_year}'] = window_bic
        bics[label] = window_bics
    return bics


def append_bigger_and_smaller(values):
    # Create a new list that will hold the original values, one smaller, and one larger values
    extended_values = []

    # Loop through the original list
    for value in values:
        # Append the original value
        extended_values.append(value)
        # Append one smaller
        extended_values.append(value - 1)
        # Append one larger
        extended_values.append(value + 1)

    return extended_values

def best_fit_noise_model(data_series, filter_frequencies=None, criteria='bic'):
    # Step 1: Remove the mean from the data series
    data_z = data_series - data_series.mean()

    # Step 2: Remove the trend using polynomial detrending (linear detrend)
    data_z_detrended = signal.detrend(data_z)

    # Step 3: Apply FFT to get the frequency components of the signal
    z_fft = rfft(data_z_detrended)
    n = len(data_z_detrended)
    timestep = 1/365.25  # Assuming daily measurements, converting to cycles per year
    freqs = rfftfreq(n, d=timestep)
    # freqs = 1/freqs
    # Define the frequencies to filter out
    if filter_frequencies is None:
        filter_frequencies = [1, 2, 3] + [1.04 * i for i in range(1, 14)] + [24.74, 26.82]
    # filter_frequencies = [365.25/x for x in filter_frequencies]
    indices = []
    for freq in filter_frequencies:
        indices.extend(list([index[0] for index, value in np.ndenumerate(freqs) if abs(value-freq) < 0.2]))

    indices = append_bigger_and_smaller(indices)
    z_fft[indices] = complex(0)

    # Reconstruct the signal by applying the inverse FFT
    z_filtered = irfft(z_fft)

    # Add the reconstructed signal to the dataframe for comparison
    residuals = z_filtered
    # Helper functions to calculate AIC and BIC for each model

    # White Noise Model (basic random noise)
    def white_noise_model(residuals):
        mean = np.mean(residuals)
        variance = np.var(residuals)
        n = len(residuals)
        log_likelihood = -n / 2 * np.log(2 * np.pi * variance) - np.sum((residuals - mean) ** 2) / (2 * variance)
        k = 2  # Two parameters: mean and variance
        aic = 2 * k - 2 * log_likelihood
        bic = np.log(n) * k - 2 * log_likelihood
        return aic, bic

    # AR(1) Model (First-order autoregressive)
    def ar1_model(residuals):
        model = ARIMA(residuals, order=(1, 0, 0))
        fitted_model = model.fit()
        return fitted_model.aic, fitted_model.bic

    # Power-law Noise Model
    def powerlaw_model(residuals):
        shape, loc, scale = powerlaw.fit(residuals)
        log_likelihood = np.sum(np.log(powerlaw.pdf(residuals, shape, loc, scale)))
        k = 3  # Three parameters: shape, loc, and scale
        n = len(residuals)
        aic = 2 * k - 2 * log_likelihood
        bic = np.log(n) * k - 2 * log_likelihood
        return aic, bic

    # Generalized Gauss-Markov Model (ARIMA(1,0,1) as approximation)
    def ggm_model(residuals):
        model = ARIMA(residuals, order=(1, 0, 1))
        fitted_model = model.fit()
        return fitted_model.aic, fitted_model.bic

    # Flicker Noise Model (1/f noise, approximated using ARIMA)
    def flicker_noise_model(residuals):
        model = ARIMA(residuals, order=(0, 1, 1))  # Approximation of 1/f noise
        fitted_model = model.fit()
        return fitted_model.aic, fitted_model.bic

    # Calculate AIC and BIC for each model
    aic_bic_results = {
        "White Noise": white_noise_model(residuals),
        "White + AR(1)": ar1_model(residuals),
        "White + Power-law Noise": powerlaw_model(residuals),
        "White + GGM": ggm_model(residuals),
        "White + Flicker Noise": flicker_noise_model(residuals)
    }

    # Find the model with the lowest AIC and BIC
    best_model_aic = min(aic_bic_results, key=lambda x: aic_bic_results[x][0])
    best_model_bic = min(aic_bic_results, key=lambda x: aic_bic_results[x][1])

    if criteria=='bic':
        return best_model_bic
    elif criteria=='aic':
        return best_model_aic

def unpack_dict_preserving_index(df):
    # Step 1: Unpack each column containing dicdf = df.replace(np.nan,{'X':np.nan, 'Y':np.nan, 'Z':np.nan})tionaries into separate columns for 'x', 'y', and 'z'
    df = df.applymap(lambda x: {'X': np.nan, 'Y': np.nan, 'Z': np.nan} if pd.isna(x) else x)
    unpack_dict = {col: pd.DataFrame(df[col].tolist(), index=df.index) for col in df.columns}
    unpacked = pd.concat(unpack_dict, axis=1)

    # Step 2: Stack the unpacked dataframe to move 'x', 'y', and 'z' into the index
    stacked = unpacked.stack(level=1)

    # Step 3: Rename the new index level to 'type'
    stacked.index.set_names('type', level=-1, inplace=True)

    return stacked
def plot_changes_in_value(data, unit, plotname):

    fig, axs = plt.subplots(3, 1, figsize=(15 / 2.54, 11 / 2.54), sharex=True,
                            dpi=300)  # Convert cm to inches by dividing by 2.54
    fig2, axs2 = plt.subplots(1, 3, figsize=(14 / 2.54, 7 / 2.54),
                            dpi=300)  # Convert cm to inches by dividing by 2.54

    for (comp, df), ax, ax2 in zip(data.groupby('type'), axs, axs2):
        # Plot heatmap using amplitudes for each coordinate
        df.index = df.index.get_level_values(0)
        df = df.T
        if unit == 'deg':
            if (comp == 'X') | (comp == 'Z'):
                df = df.applymap(lambda x: x - 360 if x > 0 else x)
        df = df.sort_index(axis=0,ascending=False)
        ax.imshow(df, aspect='auto', cmap='cividis')
        sns.boxplot(df.T,whis=(0,100),ax=ax2)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.index)
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45,ha='right')
        ax2.set_xticklabels(df.index, rotation=45,ha='right')
        ax.set_ylabel(f"{comp} [{unit}]")
        ax2.set_xlabel(f"{comp} [{unit}]")
        ax2.grid(True, axis='y', which='both', linewidth=0.8, linestyle='--', color='grey',zorder=-10)

        for (j, i), label in np.ndenumerate(df.values):
            if not np.isnan(label):
                text = ax.text(i, j, f'{label:.1f}', ha='center', va='center', fontsize=6)
                text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground='white'),
                                       path_effects.Normal()])
    fig.tight_layout(pad=.1)
    fig2.tight_layout(pad=.1)
        # plt.show()
    save_dir = './PLOTS/TS/'
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir,f"HEATMAP_{plotname}.png"))
    fig2.savefig(os.path.join(save_dir, f"BOX_{plotname}.png"))
    plt.show()

def vce_weights_boxplot(dataframes,shifts,colors):
    fig, axs = plt.subplots(1, 1, figsize=(8.4 / 2.54, 8 / 2.54), dpi=300)  # Convert cm to inches by dividing by 2.54

    # Loop through the dataframes and shifts to plot each
    for i, df in enumerate(dataframes):
        # Calculate positions for this dataframe
        positions_shifted = [x + shifts[i] for x in range(len(df.columns))]

        # Plot each dataframe with shifted positions
        sns.boxplot(data=df, width=0.2, positions=positions_shifted, color=colors[i],showfliers=False)

    # Adjust x-axis ticks to show the columns properly
    plt.xticks(ticks=[x + 0.3 for x in range(len(df.columns))], labels=df.columns)
    weights_vce = pd.read_pickle('vce_weights_0820.pkl')[df.columns]
    for (name, value),x in zip(weights_vce.items(),range(len(df.columns))):
        plt.scatter(x=x+0.3,y=value,zorder=100,marker='x',color=name_to_color(name))
    # Add a legend
    legend_labels = ["1 M", "3 M", "6 M", "1 Y"]
    handles = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(len(colors))]

    # Add the legend to the plot
    plt.legend(handles=handles, loc="upper right")
    plt.tight_layout(pad=.1)
    plt.grid(True, which='both',axis='y', linestyle='--', linewidth=0.3)
    # plt.show()
    save_dir = './PLOTS/TS/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'VCE_WEIGHTS_BOXPLOT.png'))
    plt.show()
    return fig, axs

def time_series_of_weights(weights, title):
    data = weights.copy()
    data['aa'] = 0
    data.sort_index(axis=1, inplace=True)
    data = data.fillna(0).cumsum(axis=1).astype(float)
    columns = data.columns
    data.index = data.index.to_timestamp()

    # Correct the plotting logic to properly fill between subsequent pairs of columns
    fig, axs = plt.subplots(1, 1, figsize=(8.4 / 2.54, 8 / 2.54), dpi=300)  # Convert cm to inches by dividing by 2.54

    # Loop through pairs of subsequent columns and create fill_between plots
    for i in range(len(columns) - 1):
        col1 = columns[i]
        col2 = columns[i + 1]

        # Fill the area between the two columns
        plt.fill_between(data.index, data[col1], data[col2], alpha=0.8, label=col2,
                         color = name_to_color(col2))

    # Add labels, title, and legend
    leg = axs.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=10,
                        handletextpad=0.3, columnspacing=0.2, handlelength=0.3,
                        handleheight=1.5)  # Adjust spacing and handle length/height

    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    axs.autoscale(enable=True, axis='x', tight=True)
    axs.autoscale(enable=True, axis='y', tight=True)
    plt.xlabel('Date')
    plt.ylabel(f'Weight [%] \n {title}')

    plt.tight_layout(pad=.1)
    # plt.show()
    save_dir = './PLOTS/TS/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'WEIGHTS_{title}.png'))
    plt.show()
    return fig, axs

# Function to plot FFT-based amplitude spectrum
def plot_fft_amplitude_spectrum(dataframes, offsets={'X': 2, 'Y': 2, 'Z': 2}, plotname="FFT_PLOT"):

    # Set the figure size to 8 cm x 18 cm and dpi to 300
    fig, axs = plt.subplots(3, 1, figsize=(8.4/2.54, 19/2.54), sharex=True, dpi=300)  # Convert cm to inches by dividing by 2.54

    for name, df in dataframes.items():
        n = len(df)
        time_step = 1  # Since the data is daily sampled, time_step = 1 day
        freqs = rfftfreq(n, time_step)  # Frequency in cycles per day
        mask = freqs > 0  # We only care about positive frequencies

        for i, component in enumerate(['X', 'Y', 'Z']):
            series = df[component].to_numpy()
            series = series - np.nanmean(series)
            series = np.nan_to_num(series)

            fft_vals = rfft(series)
            amplitude = 2 * np.abs(fft_vals) / n  # Amplitude spectrum

            # Apply offset depending on the component
            offset = offsets[component] * list(dataframes.keys()).index(name)
            axs[i].plot(1 / freqs[mask], amplitude[mask] + offset, label=name, color=name_to_color(name), linewidth=0.8)

    # Set y-axis labels
    axs[0].set_ylabel('Amplitude (X)')
    axs[1].set_ylabel('Amplitude (Y)')
    axs[2].set_ylabel('Amplitude (Z)')

    # Set x-axis label on the last subplot
    axs[2].set_xlabel('Period (days)')

    # Set x-scale to logarithmic and limit x-axis to periods up to 500 days
    for i, ax in enumerate(axs):
        ax.set_xscale('log')
        ax.set_xlim(1, 500)  # Show periods up to 500 days
        ax.grid(True, linestyle='--', linewidth=0.3,zorder=0)
        for i in range(1,14):
            ax.axvline(351/i, color='k',linewidth=.5,zorder=0)
        for i in range(1,4):
            ax.axvline(365.25/i, color='grey',linewidth=.7,zorder=0)

        # Set y-major ticks every `offset`
        ax.yaxis.set_major_locator(MultipleLocator(offsets[component]))

        # Set y-minor ticks every `offset / 4`
        ax.yaxis.set_minor_locator(MultipleLocator(offsets[component] / 4))

        # Enable grid for minor ticks
        ax.grid(True, which='both', linestyle='--', linewidth=0.3)

    # Set legend only on the first subplot
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=10,
                  handletextpad=0.3, columnspacing=0.1, handlelength=0.3, handleheight=1.5)  # Adjust spacing and handle length/height

    # Adjust layout and spacing
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout(pad=0.1)

    # Save the plot
    save_dir = './PLOTS/TS/'
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    plt.savefig(os.path.join(save_dir, plotname + '.png'))
    plt.show()

    # plt.show()

    return fig, axs
def export_formatted_text(df, target_directory, filename):
    """
    Exports a DataFrame to a formatted text file.

    Parameters:
    - df: The DataFrame to be exported.
    - target_directory: The directory where the text file will be saved.
    - filename: The name of the output text file.

    Returns:
    - The path to the saved file.
    """
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    # Define the format string for each line
    format_str = "{:<5} {:<3} {:<9} {:<23} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}\n"

    # Convert the DataFrame to a formatted string
    output_lines = []
    for index, row in df.iterrows():
        formatted_line = format_str.format(*row)
        output_lines.append(formatted_line)

    # Combine all lines into the final text
    formatted_text = ''.join(output_lines)

    # Full path to the output file
    output_file_path = os.path.join(target_directory, filename)

    # Save the formatted text to a file
    with open(output_file_path, 'w') as file:
        file.write(formatted_text)

    return output_file_path

# Load data and prepare dataframes
def load_and_prepare_IGN_data():
    ig3 = load_data('./DATA/IGN/ig3.gc')
    ig3 = ig3[~ig3.index.duplicated(keep='first')]  # Remove any potential duplicates

    files = ['jpl.dgc', 'mit.dgc', 'ngs.dgc', 'cod.dgc', 'esa.dgc', 'gfz.dgc', 'grg.dgc','tug.dgc','whu.dgc','ulr.dgc']
    dataframes = {file.split('.')[0]: load_data(f'./DATA/IGN/{file}') for file in files}

    # Ensure all dataframes are daily sampled and align with ig3
    date_range = pd.date_range(start=ig3.index.min(), end=ig3.index.max(), freq='D')
    for name, df in dataframes.items():
        df = df.reindex(date_range)  # Reindex to ensure daily sampling
        df = df[~df.index.duplicated(keep='first')]  # Remove any potential duplicates
        df['X'] = df['X'] + ig3['X'].reindex(date_range)
        df['Y'] = df['Y'] + ig3['Y'].reindex(date_range)
        df['Z'] = df['Z'] + ig3['Z'].reindex(date_range)
        df = df.reset_index()
        df = df[['gpsweek', "week_dow", "MJD", "index", "X", "Y", "Z", "sigmaX", "sigmaY", "sigmaZ"]]
        df = df.dropna(how='any', axis=0)
        df['index'] = df['index'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        export_formatted_text(df,r'.//DATA//IGN//',f'{name}.gc')


def plot_amplitudes_phases(data, plotname='BARS', width=14, height=10):
    components = ['X', 'Y', 'Z']
    fig, axes = plt.subplots(3, 2, figsize=(width / 2.54, height / 2.54), sharex=True, dpi=300)
    for i, component in enumerate(components):
        # Filter data by component
        component_data = data[data['Component'] == component]

        # Extract data for plotting
        datasets = component_data['Dataset']
        if 'Label' not in component_data.columns:
            component_data['Label'] = component_data['Dataset']
        dataset_labels = component_data['Label']
        amplitudes = component_data['1cpy_amp']
        amp_errors = component_data['1cpy_amp_e']
        phases = component_data['1cpy_phase']
        if (component == 'X') | (component == 'Z'):
            phases = phases.map(lambda x: x - 360 if x > 0 else x)
        phase_errors = component_data['1cpy_phase_e']

        colors = [name_to_color(ds) for ds in datasets]

        # Plot Amplitudes
        axes[i, 0].bar(dataset_labels, amplitudes, yerr=amp_errors, color=colors, capsize=5)
        axes[i, 0].set_title(f'{component} 1yr Amp')
        axes[i, 0].set_ylabel('Amplitude [mm]')
        axes[i, 0].set_xticklabels(dataset_labels, rotation=45, ha='right')
        axes[i, 0].grid(axis='y',which='both')
        axes[i, 0].autoscale(enable=True, axis='x', tight=True)

        # Plot Phases
        axes[i, 1].bar(dataset_labels, phases, yerr=phase_errors, color=colors, capsize=5)
        axes[i, 1].set_title(f'{component} 1yr Phase')
        axes[i, 1].set_ylabel('Phase [deg]')
        axes[i, 1].set_xticklabels(dataset_labels, rotation=45, ha='right')
        axes[i, 1].grid(axis='y', which='both')
        axes[i, 1].autoscale(enable=True, axis='x', tight=True)

    axes[0, 0].set_ylim(0, 4)
    axes[1, 0].set_ylim(0, 5)
    axes[2, 0].set_ylim(0, 6)

    axes[0, 1].set_ylim(-100, -180)
    axes[1, 1].set_ylim(90, 160)
    axes[2,1].set_ylim(-90,-180)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout(pad=0.1)

    # Save the plot
    save_dir = './PLOTS/TS/'
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    plt.savefig(os.path.join(save_dir, plotname + '.png'))
    plt.show()

def plot_amplitudes_phases_polar(data, plotname='POLAR', width=14, height=10):
    components = ['X', 'Y', 'Z']
    suffixes = ['FULL_','0820_']
    amp_lim, pha_lim = {}, {}
    amp_lim['X'] = (0, 4)
    amp_lim['Y'] = (0, 5)
    amp_lim['Z'] = (0, 6)

    pha_lim['X'] = (-90, -180)
    pha_lim['Y'] = (90, 180)
    pha_lim['Z'] = (-90,-180)

    for j, suffix in enumerate(suffixes):
        for i, component in enumerate(components):
            # Filter data by component
            component_data = data[data['Component'] == component]
            component_data = component_data[component_data['Label'].str.startswith(suffix)]
            fig = plt.figure(figsize=(5 / 2.54, 5 / 2.54), dpi=300)
            ax = plt.subplot(111, polar=True)
            # Set the grid and labels
            ax.set_theta_direction(-1)  # Clockwise
            ax.set_theta_offset(np.pi / 2)  # Start at 90 degrees

            ax.set_rlabel_position(90)  # Set radial labels position
            ax.grid(True, linestyle='--', linewidth=0.5)

            # Extract data for plotting
            datasets = component_data['Dataset']
            dataset_labels = component_data['Dataset']
            amplitudes = component_data['1cpy_amp']
            phases = component_data['1cpy_phase']
            phases = phases.map(lambda x: np.radians(x - 360) if x > 0 else np.radians(x))

            colors = [name_to_color(ds) for ds in datasets]
            ax.set_thetamin(min(pha_lim[component]))
            ax.set_thetamax(max(pha_lim[component]))
            ax.set_ylim(amp_lim[component][0],amp_lim[component][1])
            # ax.set_title(f'{component} {suffix.replace("_","")}')
            ax.scatter(phases, amplitudes, s=30, label=dataset_labels,
                       color=colors, marker='*', alpha=0.8)

            # ax.set_rticks([amp_lim[component][0], amp_lim[component][1]/4, amp_lim[component][1]/2, 3*amp_lim[component][1]/4, amp_lim[component][1]])

            # Save the plot
            fig.tight_layout()
            save_dir = './PLOTS/TS/'
            os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
            plt.savefig(os.path.join(save_dir, plotname + f'_{suffix}{component}.png'))
            plt.show()

def create_amp_dict(popt, pcov, key, component):
    # Extract annual and semiannual coefficients
    a0, a1, b1, a2, b2 = popt[0], popt[1], popt[2], popt[3], popt[4]

    # Get the standard deviation of the fit parameters from the covariance matrix
    perr = np.sqrt(np.diag(pcov))  # Standard errors
    a0_err, a1_err, b1_err, a2_err, b2_err = perr[0], perr[1], perr[2], perr[3], perr[4]

    # Calculate amplitudes and phases for annual and semiannual components
    annual_amplitude = np.sqrt(a1 ** 2 + b1 ** 2)
    annual_phase = np.degrees(np.arctan2(b1, a1))  # % 360  # Convert to degrees and ensure positive value
    semiannual_amplitude = np.sqrt(a2 ** 2 + b2 ** 2)
    semiannual_phase = np.degrees(np.arctan2(b2, a2))  # % 360  # Convert to degrees and ensure positive value

    # Calculate amplitude and phase errors using error propagation
    annual_amplitude_err = np.sqrt((a1 * a1_err) ** 2 + (b1 * b1_err) ** 2) / annual_amplitude
    semiannual_amplitude_err = np.sqrt((a2 * a2_err) ** 2 + (b2 * b2_err) ** 2) / semiannual_amplitude

    annual_phase_err = np.degrees(
        np.sqrt((b1_err / a1) ** 2 + (a1_err * b1 / (a1 ** 2)) ** 2) / (1 + (b1 / a1) ** 2))
    semiannual_phase_err = np.degrees(
        np.sqrt((b2_err / a2) ** 2 + (a2_err * b2 / (a2 ** 2)) ** 2) / (1 + (b2 / a2) ** 2))

    # Append results to the list
    res_dict = {
        'Dataset': key,
        'Component': component,
        'offset': a0,
        'offset_e': a0_err,
        '1cpy_amp': annual_amplitude,
        '1cpy_amp_e': annual_amplitude_err,
        '1cpy_phase': annual_phase,
        '1cpy_phase_e': annual_phase_err,
        '2cpy_amp': semiannual_amplitude,
        '2cpy_amp_e': semiannual_amplitude_err,
        '2cpy_phase': semiannual_phase,
        '2cpy_phase_e': semiannual_phase_err,
        'a1': a1,
        'b1': b1,
        'a2': a2,
        'b2': b2
    }
    return res_dict

def fit_and_provide_annual_semiannual_table_with_errors(data_dict, components=['X', 'Y', 'Z']):
    """
    Fit annual and semi-annual signals to time series data and provide results with error estimates.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing dataset names as keys and pandas DataFrames as values.
        Each DataFrame should have a datetime index and columns for the specified components.
    components : list
        List of component names to analyze (default: ['X', 'Y', 'Z'])

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing amplitude and phase information with error estimates for each dataset and component.
    """
    # List to store results
    results = []

    # Define the reference date (January 1st of the first year in the dataset)
    # This ensures phase is always calculated with respect to the beginning of the year

    for key in data_dict.keys():
        df = data_dict[key]

        # Check if there's enough data to fit the model
        if len(df) < 10:  # Minimum required points for a reasonable fit
            print(f"Warning: Dataset '{key}' has fewer than 10 points. Skipping.")
            continue

        # Get the first year in the dataset
        first_year = pd.to_datetime(df.index).year.min()
        reference_date = pd.Timestamp(f"{first_year}-01-01")

        # Convert dates to days since reference date
        t_days = (pd.to_datetime(df.index) - reference_date).days
        # Convert to years (more interpretable for annual signals)
        t_years = t_days / 365.25

        for component in components:
            # Check if component exists in the dataset
            if component not in df.columns:
                print(f"Warning: Component '{component}' not found in dataset '{key}'. Skipping.")
                continue

            # Extract the component's data
            y = df[component].values

            # Handle potential NaN values
            valid_indices = ~np.isnan(y)
            if np.sum(valid_indices) < 10:
                print(f"Warning: Not enough valid data for '{key}', component '{component}'. Skipping.")
                continue

            # Use only valid data points for fitting
            t_valid = t_years[valid_indices]
            y_valid = y[valid_indices]

            try:
                # Fit the model and calculate the covariance matrix
                popt, pcov = curve_fit(annual_semiannual_model, t_valid, y_valid,
                                       p0=[0, 1, 0, 0.5, 0],  # Initial parameter guesses
                                       maxfev=10000)  # Increase max function evaluations

                # Append results to the list
                results.append(create_amp_dict(popt, pcov, key, component))
            except (RuntimeError, ValueError) as e:
                print(f"Error fitting model for '{key}', component '{component}': {e}")

    # Convert results to a DataFrame
    if not results:
        print("No successful fits were performed.")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    return result_df

# Main function to run the analyses

    # # Perform sliding window FFT analysis on one of the components (e.g., X)
    # periods, amplitudes, phases = sliding_window_fft(dataframes['jpl'], window_size=365, step_size=30, component='X')
    #
    # # Plot the amplitude over time
    # plt.figure(figsize=(10, 6))
    # plt.plot(periods, amplitudes, label='Annual Amplitude')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude (mm)')
    # plt.title('Annual Signal Amplitude Over Time (Sliding Window FFT)')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    #
    # # Plot the phase over time
    # plt.figure(figsize=(10, 6))
    # plt.plot(periods, phases, label='Annual Phase')
    # plt.xlabel('Time')
    # plt.ylabel('Phase (radians)')
    # plt.title('Annual Signal Phase Over Time (Sliding Window FFT)')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    #
    # # Perform wavelet transform analysis on one of the components (e.g., X)
    # cwtmatr, freqs = wavelet_transform(dataframes['jpl'], component='X')
    #
    # # Plot the wavelet power spectrum
    # plt.figure(figsize=(10, 6))
    # plt.imshow(np.abs(cwtmatr),
    #            extent=[dataframes['jpl'].index.min(), dataframes['jpl'].index.max(), freqs[-1], freqs[0]],
    #            aspect='auto', cmap='jet')
    # plt.colorbar(label='Power')
    # plt.ylabel('Frequency (1/day)')
    # plt.xlabel('Time')
    # plt.title('Wavelet Power Spectrum (CWT)')
    # plt.grid(True)
    # plt.show()
    #
    # # Perform harmonic analysis over time on one of the components (e.g., X)
    # periods, amplitudes, phases = harmonic_analysis_over_time(dataframes['jpl'], window_size=365, step_size=30,
    #                                                           component='X')
    #
    # # Plot the amplitude over time
    # plt.figure(figsize=(10, 6))
    # plt.plot(periods, amplitudes, label='Annual Amplitude')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude (mm)')
    # plt.title('Annual Signal Amplitude Over Time (Harmonic Analysis)')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    #
    # # Plot the phase over time
    # plt.figure(figsize=(10, 6))
    # plt.plot(periods, phases, label='Annual Phase')
    # plt.xlabel('Time')
    # plt.ylabel('Phase (radians)')
    # plt.title('Annual Signal Phase Over Time (Harmonic Analysis)')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    #
    # # Perform time-frequency spectrogram analysis on one of the components (e.g., X)
    # freqs, times, Sxx = time_frequency_spectrogram(dataframes['jpl'], component='X')
    #
    # # Convert frequencies to periods
    # periods = 1 / freqs
    #
    # # Plot the spectrogram
    # plt.figure(figsize=(10, 6))
    # plt.pcolormesh(dataframes['jpl'].index[0] + pd.to_timedelta(times, unit='D'), periods, Sxx, shading='gouraud',
    #                cmap='jet')
    # plt.colorbar(label='Power')
    # plt.ylabel('Period (days)')
    # plt.xlabel('Time')
    # plt.ylim(0, 500)
    # plt.title('Time-Frequency Spectrogram')
    # plt.grid(True)
    # plt.show()

def geocenter_to_degree1(geocenter_df):
    """
    Convert geocenter motion time series to degree-1 spherical harmonic coefficients.

    Parameters:
    -----------
    geocenter_df : pandas.DataFrame
        DataFrame containing geocenter motion data with columns:
        'date', 'X', 'Y', 'Z', 'sigmaX', 'sigmaY', 'sigmaZ'
        where X, Y, Z are in millimeters.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: 'date', 'date_str_x', 'C10', 'C10_error',
        'date_str_y', 'C11', 'S11', 'C11_error', 'S11_error'
        where C10, C11, S11 are expressed in kg/m²
    """
    # Constants
    a = 6371000.0  # Earth's radius in meters
    rho_e = 5517.0  # Earth's average density in kg/m³

    # Create a copy of the input dataframe with only the required columns

    geocenter_df = geocenter_df.reset_index()
    df = geocenter_df[['ISO_date', 'X', 'Y', 'Z', 'sigmaX', 'sigmaY', 'sigmaZ']].copy()

    # Convert X, Y, Z from mm to m
    for col in ['X', 'Y', 'Z', 'sigmaX', 'sigmaY', 'sigmaZ']:
        df[col] = df[col]

    # Calculate the degree-1 coefficients

    RHO_E = 5517.0  # Average density of Earth in kg/m³
    K1_CF = 0.021  # Load Love number for degree 1

    factor = - RHO_E / (1 + K1_CF) / 1000.0

    df['C11'] = df['X'] * factor
    df['S11'] = df['Y'] * factor
    df['C10'] = df['Z'] * factor

    # Calculate errors
    df['C11_error'] = df['sigmaX'] * abs(factor)
    df['S11_error'] = df['sigmaY'] * abs(factor)
    df['C10_error'] = df['sigmaZ'] * abs(factor)

    df['ISO_date'] = df['ISO_date'].dt.tz_localize(None)

    # Select and reorder columns for the output
    result_df = df[['ISO_date', 'C10', 'C10_error',
                    'C11', 'S11', 'C11_error', 'S11_error','X','sigmaX','Y','sigmaY','Z','sigmaZ']]

    result_df.rename({'ISO_date':'date',
                      'sigmaX':'X_error',
                      'sigmaY':'Y_error',
                      'sigmaZ':'Z_error'},axis=1,inplace=True)

    return result_df.set_index('date')