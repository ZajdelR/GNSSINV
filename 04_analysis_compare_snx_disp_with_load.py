import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from toolbox_gfzload2pkl import combine_selected_files
import os
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
matplotlib.use('TkAgg')

# Load data
sta = 'NAUS'

df = pd.read_pickle(f'INPUT_CRD/IGS1R03SNX_07D/CODE/IGS1R03SNX_{sta}_07D_DISP.PKL')
df = df.reset_index(level='EPOCH')[['EPOCH','dU','dN','dE']].set_index('EPOCH')

h = pd.read_pickle(f'SOLUTION_PICKLES_GFZ_IGS1R03/{sta}_H_cf.PKL')
h = h.rename({'R':'dU',
              'NS':'dN',
              'EW':'dE'},axis=1)

files = [f'SOLUTION_PICKLES_GFZ_IGS1R03/{sta}_A_cf.PKL',
         f'SOLUTION_PICKLES_GFZ_IGS1R03/{sta}_O_cf.PKL',
         f'SOLUTION_PICKLES_GFZ_IGS1R03/{sta}_S_cf.PKL']

files_df = {os.path.basename(x):pd.read_pickle(x) for x in files}

aos, name = combine_selected_files(files_df)
aos = aos.rename({'R':'dU',
                  'NS':'dN',
                  'EW':'dE'},axis=1)

# Convert indices to date objects for consistency
df.index = df.index.date
aos.index = aos.index.date
h.index = h.index.date
df_red = (df - aos).dropna()

# Find common dates between df and h
common_dates = df.index.intersection(h.index)
print(f"Found {len(common_dates)} common dates between the df and h datasets.")

# Filter to common dates
df_common = df.loc[common_dates]
h_common = h.loc[common_dates]

# Calculate differences for dU
differences_dU = df_common['dU'] - h_common['dU']

# Create a figure with 2 rows, 1 column for dU comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Upper subplot: raw time series (only common dates)
axes[0].plot(df_common.index, df_common['dU'], 'o-', label='df dU', color='blue', alpha=0.7, markersize=3)
axes[0].plot(h_common.index, h_common['dU'], 'x-', label='h dU', color='red', alpha=0.7, markersize=3)
axes[0].set_ylabel('dU Value (mm)')
axes[0].set_title(f'Time Series Comparison of dU for {sta} - Raw Data (Common Dates)')
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(loc='best')

# Lower subplot: differences
axes[1].plot(common_dates, differences_dU, 'o-', color='purple', alpha=0.8, markersize=3)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].fill_between(common_dates, differences_dU, 0, where=differences_dU>0, color='green', alpha=0.3)
axes[1].fill_between(common_dates, differences_dU, 0, where=differences_dU<0, color='red', alpha=0.3)
axes[1].set_ylabel('Difference (df - h)')
axes[1].set_title('Differences Between df and h for dU')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].set_xlabel('Date')

# Improve x-axis formatting
fig.autofmt_xdate()

# Compute statistics on the differences
mean_diff = differences_dU.mean()
std_diff = differences_dU.std()
median_diff = differences_dU.median()
max_diff = differences_dU.max()
min_diff = differences_dU.min()
rms_diff = np.sqrt(np.mean(differences_dU**2))
corr, p_value = stats.pearsonr(df_common['dU'], h_common['dU'])

# Add text box with statistics
stats_text = (
    f'Statistics of consistency:\n'
    f'Mean difference: {mean_diff:.4f} mm\n'
    f'Median difference: {median_diff:.4f} mm\n'
    f'Std deviation: {std_diff:.4f} mm\n'
    f'RMS difference: {rms_diff:.4f} mm\n'
    f'Min/Max diff: {min_diff:.4f}/{max_diff:.4f} mm\n'
    f'Correlation: {corr:.4f} (p={p_value:.2e})'
)
# Add statistics as text box in the bottom panel
axes[1].text(1.02, 0.5, stats_text, transform=axes[1].transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='center')

# Create a histogram of differences in a separate figure
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.hist(differences_dU, bins=20, alpha=0.7, color='purple')
ax2.axvline(x=0, color='k', linestyle='--')
ax2.axvline(x=mean_diff, color='r', linestyle='-', label=f'Mean: {mean_diff:.4f}')
ax2.axvline(x=median_diff, color='g', linestyle='-', label=f'Median: {median_diff:.4f}')
ax2.set_xlabel('Difference (df - h) for dU')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Distribution of Differences in dU for {sta}')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

# Optional: Create a scatter plot to check for correlation
fig3, ax3 = plt.subplots(figsize=(8, 8))
ax3.scatter(df_common['dU'], h_common['dU'], alpha=0.6)
# Add a diagonal line (perfect correlation)
min_val = min(df_common['dU'].min(), h_common['dU'].min())
max_val = max(df_common['dU'].max(), h_common['dU'].max())
ax3.plot([min_val, max_val], [min_val, max_val], 'k--')
ax3.set_xlabel('df dU values')
ax3.set_ylabel('h dU values')
ax3.set_title(f'Correlation Plot (r={corr:.4f})')
ax3.grid(True, linestyle='--', alpha=0.6)
# Add correlation info
ax3.text(0.05, 0.95, f'Correlation: {corr:.4f}\np-value: {p_value:.2e}',
         transform=ax3.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust layout and display plots
plt.tight_layout()
plt.show()

# Print basic statistics for reference
print("\nStatistics of consistency between df and h for dU:")
print(f"Mean difference: {mean_diff:.4f} mm")
print(f"Median difference: {median_diff:.4f} mm")
print(f"Standard deviation: {std_diff:.4f} mm")
print(f"RMS difference: {rms_diff:.4f} mm")
print(f"Min/Max difference: {min_diff:.4f}/{max_diff:.4f} mm")
print(f"Pearson correlation: {corr:.4f} (p-value: {p_value:.2e})")
print(f"Number of common data points: {len(common_dates)}")