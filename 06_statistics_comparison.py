import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import matplotlib
from matplotlib.colors import TwoSlopeNorm

# Use TkAgg backend for interactive plots
matplotlib.use('TkAgg')


def analyze_and_map_comparison(file1, file2, column_name='correlation', higher_is_better=True,
                               file1_label='M', file2_label='L', output_dir=None,
                               figsize_map=(14, 10), figsize_pie=(10, 7),
                               dpi=300, cmap='plasma', plot_title=None):
    """
    Analyze two CSV files, create a map similar to create_correlation_map, and create a pie chart.

    Parameters:
    file1, file2 (str): Paths to CSV files
    column_name (str): Column to compare
    higher_is_better (bool): Whether higher values are better
    file1_label, file2_label (str): Labels for the files
    output_dir (str): Directory to save output files
    figsize_map, figsize_pie (tuple): Figure sizes for map and pie chart
    dpi (int): DPI for saved figures
    cmap (str): Colormap for the map
    plot_title (str): Custom title for plots

    Returns:
    dict: Dictionary with statistics
    """
    # Create output directory if specified and doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Read the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check if column exists in both files
    for df, file_name in [(df1, file1), (df2, file2)]:
        if column_name not in df.columns:
            raise ValueError(
                f"Column '{column_name}' not found in {file_name}. Available columns: {', '.join(df.columns)}")

    # Find station column
    station_column = None
    station_candidates = ['station', 'name', 'id', 'site', 'Station', 'Name', 'ID', 'Site']

    for candidate in station_candidates:
        if candidate in df1.columns and candidate in df2.columns:
            station_column = candidate
            break

    if station_column is None:
        print("Warning: No matching station column found. Using row index to match stations.")
        if len(df1) != len(df2):
            print(f"Warning: Files have different number of rows ({len(df1)} vs {len(df2)})")
            print("Using only the first min(len(df1), len(df2)) rows")
            min_rows = min(len(df1), len(df2))
            df1 = df1.iloc[:min_rows]
            df2 = df2.iloc[:min_rows]
    else:
        print(f"Using '{station_column}' to match stations")
        # Check if all stations match
        if not set(df1[station_column]) == set(df2[station_column]):
            print("Warning: Station sets don't match exactly between files!")
            print(f"Stations in {file1} only: {len(set(df1[station_column]) - set(df2[station_column]))}")
            print(f"Stations in {file2} only: {len(set(df2[station_column]) - set(df1[station_column]))}")

            # Use only common stations
            common_stations = set(df1[station_column]).intersection(set(df2[station_column]))
            print(f"Using {len(common_stations)} common stations")

            df1 = df1[df1[station_column].isin(common_stations)]
            df2 = df2[df2[station_column].isin(common_stations)]

            # Reindex based on station
            df1.set_index(station_column, inplace=True)
            df2.set_index(station_column, inplace=True)
            df1 = df1.reindex(sorted(common_stations))
            df2 = df2.reindex(sorted(common_stations))
            df1.reset_index(inplace=True)
            df2.reset_index(inplace=True)

    # Find lat/lon columns
    lat_candidates = ['lat', 'latitude', 'Lat', 'Latitude', 'LAT', 'LATITUDE']
    lon_candidates = ['lon', 'longitude', 'Lon', 'Longitude', 'LON', 'LONGITUDE', 'long', 'Long', 'LONG']

    lat_col = None
    lon_col = None

    for candidate in lat_candidates:
        if candidate in df1.columns:
            lat_col = candidate
            break

    for candidate in lon_candidates:
        if candidate in df1.columns:
            lon_col = candidate
            break

    if not lat_col or not lon_col:
        print("Error: Latitude or longitude columns not found in data. Map cannot be created.")
        return None

    # Compare values
    if higher_is_better:
        file1_better = sum(df1[column_name] > df2[column_name])
        file2_better = sum(df2[column_name] > df1[column_name])
        equal = sum(df1[column_name] == df2[column_name])
    else:
        file1_better = sum(df1[column_name] < df2[column_name])
        file2_better = sum(df2[column_name] < df1[column_name])
        equal = sum(df1[column_name] == df2[column_name])

    # Create a new column indicating which solution is better
    if higher_is_better:
        df1['better_solution'] = np.where(df1[column_name] > df2[column_name], file1_label,
                                          np.where(df1[column_name] < df2[column_name], file2_label, 'Equal'))
    else:
        df1['better_solution'] = np.where(df1[column_name] < df2[column_name], file1_label,
                                          np.where(df1[column_name] > df2[column_name], file2_label, 'Equal'))

    # Calculate differential
    df1['diff'] = df1[column_name] - df2[column_name]

    # Calculate percentage improvement
    # For higher_is_better metrics, negative percentage means file1 (M) is better
    # For lower_is_better metrics, positive percentage means file1 (M) is better
    # We'll adjust the sign based on the higher_is_better flag to maintain consistency
    # where negative means file1 (M) is better and positive means file2 (L) is better

    # To avoid division by zero, use the maximum of the two values for normalization
    max_vals = np.maximum(df1[column_name], df2[column_name])

    # Avoid division by zero for max_vals that are zero
    max_vals = np.where(max_vals == 0, 1, max_vals)

    # Calculate percentage difference
    # We reverse the sign if higher_is_better is True
    sign_factor = -1 if higher_is_better else 1
    df1['pct_improvement'] = sign_factor * (df1[column_name] - df2[column_name]) / max_vals * 100

    # Print statistics
    print(f"Total stations: {len(df1)}")
    print(f"\n{column_name.capitalize()} comparison ({('higher' if higher_is_better else 'lower')} is better):")
    print(f"{file1_label} better: {file1_better} stations ({file1_better / len(df1) * 100:.1f}%)")
    print(f"{file2_label} better: {file2_better} stations ({file2_better / len(df1) * 100:.1f}%)")
    print(f"Equal: {equal} stations ({equal / len(df1) * 100:.1f}%)")

    # Calculate average differences
    mean_diff = (df1[column_name] - df2[column_name]).mean()
    median_diff = (df1[column_name] - df2[column_name]).median()
    print(f"\nAverage {column_name} difference ({file1_label}-{file2_label}): {mean_diff:.4f}")
    print(f"Median {column_name} difference ({file1_label}-{file2_label}): {median_diff:.4f}")

    # Calculate average percentage improvement
    mean_pct_improvement = df1['pct_improvement'].mean()
    median_pct_improvement = df1['pct_improvement'].median()
    print(f"\nAverage percentage improvement: {mean_pct_improvement:.2f}% (negative means {file1_label} is better)")
    print(f"Median percentage improvement: {median_pct_improvement:.2f}% (negative means {file1_label} is better)")

    # Extract scenario and model info from filenames
    scenario = os.path.basename(file1).split('_data_')[1].split('_VS_')[0]
    model1 = os.path.basename(file1).split('_VS_')[1].split('.')[0]
    model2 = os.path.basename(file2).split('_VS_')[1].split('.')[0]

    # Prepare statistics dictionary
    stats = {
        'scenario': scenario,
        'column': column_name,
        'file1': os.path.basename(file1),
        'file2': os.path.basename(file2),
        'model1': model1,
        'model2': model2,
        'total_stations': len(df1),
        f'{file1_label}_better_count': file1_better,
        f'{file2_label}_better_count': file2_better,
        'equal_count': equal,
        f'{file1_label}_better_percent': file1_better / len(df1) * 100,
        f'{file2_label}_better_percent': file2_better / len(df1) * 100,
        'equal_percent': equal / len(df1) * 100,
        'mean_difference': mean_diff,
        'median_difference': median_diff,
        'mean_pct_improvement': mean_pct_improvement,
        'median_pct_improvement': median_pct_improvement,
        f'{file1_label}_mean': df1[column_name].mean(),
        f'{file2_label}_mean': df2[column_name].mean(),
        f'{file1_label}_median': df1[column_name].median(),
        f'{file2_label}_median': df2[column_name].median(),
    }

    # Generate title if not provided
    if not plot_title:
        plot_title = f"{column_name.capitalize()} Comparison: {scenario} - {file1_label} vs {file2_label}"

    # 1. Create Map Plot with Cartopy (which solution is better)
    fig_map = plt.figure(figsize=figsize_map)
    ax_map = fig_map.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # Add map features
    ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

    # Set up gridlines
    gl = ax_map.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.MaxNLocator(nbins=6)
    gl.ylocator = mticker.MaxNLocator(nbins=6)

    ax_map.set_global()

    # Plot the comparison (which solution is better)
    color_map = {file1_label: 'blue', file2_label: 'orange', 'Equal': 'green'}

    # Get colors for the comparison plot
    comparison_colors = [color_map[sol] for sol in df1['better_solution']]

    # Plot station points based on which solution is better
    scatter_comparison = ax_map.scatter(
        df1[lon_col], df1[lat_col],
        transform=ccrs.PlateCarree(),
        c=comparison_colors,
        s=50,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
        zorder=10
    )

    # Create legend elements for comparison plot
    legend_elements = []
    if file1_better > 0:
        legend_elements.append(Patch(facecolor='blue', edgecolor='black', alpha=0.7,
                                     label=f'{file1_label} Better ({file1_better} stations)'))
    if file2_better > 0:
        legend_elements.append(Patch(facecolor='orange', edgecolor='black', alpha=0.7,
                                     label=f'{file2_label} Better ({file2_better} stations)'))
    if equal > 0:
        legend_elements.append(Patch(facecolor='green', edgecolor='black', alpha=0.7,
                                     label=f'Equal ({equal} stations)'))

    ax_map.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Add title and statistics
    ax_map.set_title(f"{plot_title}\nBetter solution at each station", fontsize=14)

    stats_text = (
        f"Total stations: {len(df1)}\n"
        f"{file1_label} better: {file1_better} ({file1_better / len(df1) * 100:.1f}%)\n"
        f"{file2_label} better: {file2_better} ({file2_better / len(df1) * 100:.1f}%)\n"
        f"Equal: {equal} ({equal / len(df1) * 100:.1f}%)\n"
        f"Mean diff ({file1_label}-{file2_label}): {mean_diff:.4f}"
    )

    plt.figtext(0.02, 0.15, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()

    # Save map figure
    if output_dir:
        map_file = os.path.join(output_dir, f"{scenario}_{column_name}_map_{file1_label}_vs_{file2_label}.png")
        plt.savefig(map_file, dpi=dpi, bbox_inches='tight')
        print(f"Map figure saved to {map_file}")

    # 2. Create a new map showing percentage improvement
    fig_pct = plt.figure(figsize=figsize_map)
    ax_pct = fig_pct.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # Add map features
    ax_pct.add_feature(cfeature.LAND, facecolor='lightgray')
    ax_pct.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax_pct.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax_pct.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

    # Set up gridlines
    gl = ax_pct.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.MaxNLocator(nbins=6)
    gl.ylocator = mticker.MaxNLocator(nbins=6)

    ax_pct.set_global()

    # Set fixed colorbar limits based on actual data range
    # Get actual min/max for setting reasonable colorbar limits
    actual_min = df1['pct_improvement'].min()
    actual_max = df1['pct_improvement'].max()

    # Calculate reasonable vmin and vmax based on data
    # Use the max absolute value to create a symmetric range, but cap at ±15%
    max_abs_val = min(15, max(abs(actual_min), abs(actual_max)))
    vmin = -max_abs_val
    vmax = max_abs_val

    # Use a diverging colormap with white at zero
    cmap_name = 'RdBu_r'  # Red-Blue reversed (negative: blue, positive: red)

    # Set up normalization for the colormap
    # Use TwoSlopeNorm to center the colormap at zero with fixed limits
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Plot the percentage improvement
    scatter_pct = ax_pct.scatter(
        df1[lon_col], df1[lat_col],
        transform=ccrs.PlateCarree(),
        c=df1['pct_improvement'],
        s=50,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
        zorder=10,
        cmap=cmap_name,
        norm=norm
    )

    # Add colorbar with fixed limits - FIXED VERSION
    cbar = plt.colorbar(scatter_pct, ax=ax_pct, orientation='horizontal', pad=0.05, shrink=0.75)
    cbar.set_label(f'Percentage Improvement (%) - Negative: {file1_label} better, Positive: {file2_label} better')

    # Set appropriate tick marks based on the range
    if max_abs_val <= 5:
        ticks = [-5, -2.5, 0, 2.5, 5]
    elif max_abs_val <= 10:
        ticks = [-10, -5, 0, 5, 10]
    elif max_abs_val <= 15:
        ticks = [-15, -10, -5, 0, 5, 10, 15]
    elif max_abs_val <= 25:
        ticks = [-25, -15, -5, 0, 5, 15, 25]
    elif max_abs_val <= 50:
        ticks = [-50, -25, 0, 25, 50]
    else:
        ticks = [-100, -50, 0, 50, 100]

    cbar.set_ticks(ticks)

    # Make sure the figure draws before adding more elements
    plt.draw()

    # Add title
    ax_pct.set_title(f"{plot_title}\nPercentage Improvement at each station", fontsize=14)

    stats_text = (
        f"Total stations: {len(df1)}\n"
        f"Mean % improvement: {mean_pct_improvement:.2f}%\n"
        f"Median % improvement: {median_pct_improvement:.2f}%\n"
        f"Actual range: [{actual_min:.2f}%, {actual_max:.2f}%]\n"
        f"Colorbar range: [{vmin}%, {vmax}%]\n"
        f"Negative values: {file1_label} better, Positive values: {file2_label} better"
    )

    plt.figtext(0.02, 0.05, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()

    # Save percentage improvement map
    if output_dir:
        pct_map_file = os.path.join(output_dir,
                                    f"{scenario}_{column_name}_pct_improvement_{file1_label}_vs_{file2_label}.png")
        plt.savefig(pct_map_file, dpi=dpi, bbox_inches='tight')
        print(f"Percentage improvement map saved to {pct_map_file}")

    # 3. Create Pie Chart
    fig_pie = plt.figure(figsize=figsize_pie)
    ax_pie = fig_pie.add_subplot(1, 1, 1)

    # Prepare pie chart data
    labels = [f'{file1_label} Better', f'{file2_label} Better', 'Equal']
    sizes = [file1_better, file2_better, equal]
    colors = ['#0088FE', '#FF8042', '#FFBB28']

    # Remove zero values and corresponding labels
    cleaned = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
    if cleaned:
        sizes, labels, colors = zip(*cleaned)

    # Plot pie chart
    wedges, texts, autotexts = ax_pie.pie(
        sizes,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )

    # Customize autotexts
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_weight('bold')

    # Add legend and title
    ax_pie.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
    ax_pie.set_title(f"{column_name.capitalize()} Comparison: {scenario}\n{file1_label} vs {file2_label}", fontsize=14)

    # Add statistics
    stats_text = (
        f"Total stations: {len(df1)}\n"
        f"Mean {file1_label}: {df1[column_name].mean():.4f}\n"
        f"Mean {file2_label}: {df2[column_name].mean():.4f}\n"
        f"Mean diff: {mean_diff:.4f}\n"
        f"Mean % improvement: {mean_pct_improvement:.2f}%"
    )

    plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()

    # Save pie chart
    if output_dir:
        pie_file = os.path.join(output_dir, f"{scenario}_{column_name}_pie_{file1_label}_vs_{file2_label}.png")
        plt.savefig(pie_file, dpi=dpi, bbox_inches='tight')
        print(f"Pie chart saved to {pie_file}")

    # 4. Save combined dataset with comparison results for further analysis
    if output_dir:
        # Merge dataframes to have both values in one
        if station_column:
            df_combined = pd.merge(df1, df2, on=station_column, suffixes=(f'_{file1_label}', f'_{file2_label}'))
        else:
            # If no station column, merge based on index (position)
            df2_renamed = df2.copy()
            df2_renamed.columns = [f"{col}_{file2_label}" if col != station_column else col for col in df2.columns]
            df1_renamed = df1.copy()
            df1_renamed.columns = [f"{col}_{file1_label}" if col != station_column else col for col in df1.columns]
            df_combined = pd.concat([df1_renamed, df2_renamed], axis=1)

        # Add the comparison column
        df_combined['better_solution'] = df1['better_solution']
        df_combined['diff'] = df1['diff']
        df_combined['pct_improvement'] = df1['pct_improvement']

        # Save the combined dataset
        combined_file = os.path.join(output_dir,
                                     f"{scenario}_{column_name}_combined_{file1_label}_vs_{file2_label}.csv")
        df_combined.to_csv(combined_file, index=False)
        print(f"Combined data saved to {combined_file}")

    plt.close('all')
    return stats


def run_batch_analysis(base_dir, scenarios, statistics, model1_label='LSDM', model2_label='Lisflood', output_dir=None,
                       window='_30d_400d'):
    """
    Run analysis for multiple scenarios and statistics, saving results to a CSV.

    Parameters:
    base_dir (str): Base directory containing CSV files
    scenarios (list): List of scenarios to analyze (e.g., ['WO-A', 'WO-AOS'])
    statistics (list): List of statistics to analyze (e.g., ['correlation'])
    model1_label, model2_label (str): Labels for the models
    output_dir (str): Directory to save output files

    Returns:
    pd.DataFrame: DataFrame with all statistics
    """
    if output_dir is None:
        output_dir = os.path.join(base_dir, "ANALYSIS_RESULTS")

    os.makedirs(output_dir, exist_ok=True)

    all_stats = []

    for scenario in scenarios:
        for statistic in statistics:
            print(f"\nAnalyzing {scenario} - {statistic}")

            file1 = f"ITRF2020-IGS-RES_{statistic}_data_{scenario}_VS_M_BP{window}.csv"
            file2 = f"ITRF2020-IGS-RES_{statistic}_data_{scenario}_VS_L_BP{window}.csv"

            filepath1 = os.path.join(base_dir, file1)
            filepath2 = os.path.join(base_dir, file2)

            if not os.path.exists(filepath1) or not os.path.exists(filepath2):
                print(f"Warning: Files not found for {scenario} - {statistic}")
                continue

            try:
                stats = analyze_and_map_comparison(
                    filepath1,
                    filepath2,
                    column_name=statistic,
                    higher_is_better=True,
                    file1_label=model1_label,
                    file2_label=model2_label,
                    output_dir=output_dir
                )

                if stats:
                    all_stats.append(stats)
            except Exception as e:
                print(f"Error analyzing {scenario} - {statistic}: {str(e)}")

    # Create a DataFrame with all statistics and save it
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_file = os.path.join(output_dir, f"all_comparison_statistics_{model1_label}_vs_{model2_label}.csv")
        stats_df.to_csv(stats_file, index=False)
        print(f"\nAll statistics saved to {stats_file}")
        return stats_df
    else:
        print("No statistics were collected")
        return None


def main():
    parser = argparse.ArgumentParser(description='Compare and map statistics between two models across scenarios')
    parser.add_argument('--base-dir', type=str, required=True,
                        help='Base directory containing CSV files')
    parser.add_argument('--scenarios', type=str, nargs='+', default=['WO-A', 'WO-AOS', 'WO-None'],
                        help='Scenarios to analyze')
    parser.add_argument('--statistics', type=str, nargs='+',
                        default=['correlation', 'variance_explained', 'std_reduction', 'kge2012'],
                        help='Statistics to analyze')
    parser.add_argument('--model1-label', type=str, default='LSDM',
                        help='Label for model 1 (M in filenames)')
    parser.add_argument('--model2-label', type=str, default='Lisflood',
                        help='Label for model 2 (L in filenames)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output files')

    args = parser.parse_args()

    run_batch_analysis(
        args.base_dir,
        args.scenarios,
        args.statistics,
        args.model1_label,
        args.model2_label,
        args.output_dir
    )


if __name__ == "__main__":
    # Example direct execution
    inp_dir = r'OUTPUT\SNX_LOAD_COMPARISONS\ITRF2020-IGS-RES_01D\MAPS\WITH_BP'
    scenarios = ['WO-A']
    statistics = ['kge2012', 'kge_gamma', 'std_reduction', 'variance_explained', 'correlation']
    window = '_30d_400d'
    run_batch_analysis(inp_dir, scenarios, statistics, 'LSDM', 'Lisflood', output_dir=inp_dir, window=window)