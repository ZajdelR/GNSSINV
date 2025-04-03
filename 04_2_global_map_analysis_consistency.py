import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.colors as colors
import os
import glob
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker

plt.ioff()

def load_station_results(comp_dir, pattern='*_WO-*_VS_*.PKL', exclude_pattern='*SUMMARY.PKL'):
    all_files = glob.glob(os.path.join(comp_dir, pattern))

    if exclude_pattern:
        exclude_files = set(glob.glob(os.path.join(comp_dir, exclude_pattern)))
        comparison_files = [f for f in all_files if f not in exclude_files]
    else:
        comparison_files = all_files

    if not comparison_files:
        print(f"No comparison files found in {comp_dir} matching pattern {pattern}")
        return None, None, None, None

    print(f"Found {len(comparison_files)} station comparison files")

    sum_components = None
    compare_with = None
    solution = None
    compiled_data = {}

    for file_path in comparison_files:
        try:
            filename = os.path.basename(file_path)
            parts = filename.split('_')

            if solution is None and len(parts) > 0:
                solution = parts[0]

            if len(parts) >= 3:
                sta = parts[1]
                data = pd.read_pickle(file_path)
                compiled_data[sta] = data

                if sum_components is None and 'sum_components' in data:
                    sum_components = data['sum_components']

                if compare_with is None and 'compare_with' in data:
                    compare_with = data['compare_with']

                print(f"Loaded data for station {sta}")

        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")

    return compiled_data, sum_components, compare_with, solution


def extract_statistics_for_mapping(compiled_data, min_num_points=30, min_h_std=0.0):
    mapping_stats = {}
    stations_skipped_points = 0
    stations_skipped_h_std = 0
    filtered_out_stations = {}

    for sta, data in compiled_data.items():
        try:
            stats = data['stats']
            std_devs = data['standard_deviations']
            num_points = stats['num_points'] if 'num_points' in stats else 0

            h_std = None
            if 'individual_components' in std_devs and 'H' in std_devs['individual_components']:
                h_std = std_devs['individual_components']['H']
            else:
                compare_with = data.get('compare_with', '')
                if compare_with == 'H':
                    h_std = std_devs.get('comparison_component', 0.0)

            station_data = {
                'variance_explained': stats['variance_explained'],
                'correlation': stats['correlation'],
                'kge2012': stats['kge_modified'],
                'rms': stats['rms'],
                'mean_diff': stats['mean'],
                'median_diff': stats['median'],
                'std_diff': stats['std'],
                'min_diff': stats['min'],
                'max_diff': stats['max'],
                'num_points': num_points,
                'std_original': std_devs['original_df'],
                'std_reduced': std_devs['df_reduced'],
                'std_comparison': std_devs['comparison_component'],
                'std_components': std_devs['individual_components'],
                'h_std': h_std
            }

            if num_points < min_num_points:
                filtered_out_stations[sta] = {
                    'reason': 'not_enough_points',
                    'value': num_points,
                    'threshold': min_num_points,
                    **station_data
                }
                stations_skipped_points += 1
                continue

            if h_std is not None and h_std < min_h_std:
                filtered_out_stations[sta] = {
                    'reason': 'h_std_too_low',
                    'value': h_std,
                    'threshold': min_h_std,
                    **station_data
                }
                stations_skipped_h_std += 1
                continue

            mapping_stats[sta] = station_data

        except KeyError as e:
            print(f"Missing key in data for station {sta}: {str(e)}")
        except Exception as e:
            print(f"Error processing station {sta}: {str(e)}")

    print(f"Filtered out {stations_skipped_points} stations with less than {min_num_points} data points")
    print(f"Filtered out {stations_skipped_h_std} stations with H std less than {min_h_std} mm")
    print(f"Remaining stations: {len(mapping_stats)}")

    return mapping_stats, filtered_out_stations


def visualize_variance_explained_map(comp_dir, output_dir=None, pattern='*_WO-*_VS_*.PKL', min_num_points=30,
                                     min_h_std=0.0,
                                     figsize=(14, 10), dpi=300, cmap='viridis'):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(comp_dir), "MAPS")

    compiled_data, sum_components, compare_with, solution = load_station_results(comp_dir, pattern=pattern)

    if not compiled_data:
        print("No data available for visualization")
        return None, None

    mapping_stats, filtered_out_stations = extract_statistics_for_mapping(compiled_data, min_num_points, min_h_std)

    if not mapping_stats:
        print("No stations meet the criteria for visualization")
        return None, None

    print(f"Using {len(mapping_stats)} stations after filtering")
    print(f"Excluded {len(filtered_out_stations)} stations")

    try:
        station_coords = pd.read_pickle('EXT/PROCESSINS_SUPPLEMENTS/ALL_STATIONS_LATLON.PKL')
        print(f"Loaded coordinates for {len(station_coords)} stations")
        print(f"Coordinate columns: {station_coords.columns.tolist()}")
    except Exception as e:
        print(f"Error loading station coordinates: {str(e)}")
        return None, None

    plot_data = []
    for sta, stats in mapping_stats.items():
        if sta in station_coords.index:
            lat = station_coords.loc[sta, 'Latitude']
            lon = station_coords.loc[sta, 'Longitude']

            if not np.isnan(lat) and not np.isnan(lon):
                plot_data.append({
                    'station': sta,
                    'lat': lat,
                    'lon': lon,
                    **stats
                })
            else:
                print(f"Warning: Invalid coordinates for station {sta}")
        else:
            print(f"Warning: No coordinates found for station {sta}")

    excluded_data = []
    for sta, stats in filtered_out_stations.items():
        if sta in station_coords.index:
            lat = station_coords.loc[sta, 'Latitude']
            lon = station_coords.loc[sta, 'Longitude']

            if not np.isnan(lat) and not np.isnan(lon):
                excluded_data.append({
                    'station': sta,
                    'lat': lat,
                    'lon': lon,
                    'reason': stats['reason'],
                    **stats
                })

    if not plot_data:
        print("No stations with valid coordinates found")
        return None, None

    plot_df = pd.DataFrame(plot_data)
    excluded_df = pd.DataFrame(excluded_data) if excluded_data else pd.DataFrame()

    excluded_by_points = sum(1 for _, row in excluded_df.iterrows() if row['reason'] == 'not_enough_points')
    excluded_by_h_std = sum(1 for _, row in excluded_df.iterrows() if row['reason'] == 'h_std_too_low')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.MaxNLocator(nbins=6)
    gl.ylocator = mticker.MaxNLocator(nbins=6)

    ax.set_global()

    if not excluded_df.empty:
        ax.scatter(
            excluded_df['lon'], excluded_df['lat'],
            transform=ccrs.PlateCarree(),
            color='grey',
            s=15,
            alpha=0.5,
            edgecolor=None,
            zorder=5,
            label='Excluded stations'
        )

    scatter = ax.scatter(
        plot_df['lon'], plot_df['lat'],
        transform=ccrs.PlateCarree(),
        c=plot_df['variance_explained'],
        cmap=cmap,
        vmin=0,
        vmax=100,
        s=50,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
        zorder=10,
        label='Included stations'
    )

    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label(f'Variance Explained by {compare_with} (%)')

    top_n = 15
    top_stations = plot_df.nlargest(top_n, 'variance_explained')

    for _, row in top_stations.iterrows():
        ax.text(row['lon'], row['lat'] + 2, row['station'],
                transform=ccrs.PlateCarree(),
                fontsize=8, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', linewidth=0))

    title = f'Variance in {solution} Explained by {compare_with} (without {sum_components})'
    subtitle = f'Stations with at least {min_num_points} data points and H std ≥ {min_h_std} mm'
    ax.set_title(f'{title}\n{subtitle}', fontsize=14)

    stats_text = (
        f"Total stations: {len(plot_df)}\n"
        f"Excluded stations: {len(excluded_df)} (Points: {excluded_by_points}, H std: {excluded_by_h_std})\n"
        f"Mean variance explained: {plot_df['variance_explained'].mean():.2f}%\n"
        f"Median variance explained: {plot_df['variance_explained'].median():.2f}%\n"
        f"Range: {plot_df['variance_explained'].min():.2f}% - {plot_df['variance_explained'].max():.2f}%"
    )

    plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir,
                                   f'{solution}_variance_explained_map_WO-{sum_components}_VS_{compare_with}.png')
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

        csv_file = os.path.join(output_dir,
                                f'{solution}_variance_explained_data_WO-{sum_components}_VS_{compare_with}.csv')
        plot_df.to_csv(csv_file, index=False)
        print(f"Data saved to {csv_file}")

        if not excluded_df.empty:
            excluded_csv = os.path.join(output_dir,
                                        f'{solution}_excluded_stations_WO-{sum_components}_VS_{compare_with}.csv')
            excluded_df.to_csv(excluded_csv, index=False)
            print(f"Excluded stations data saved to {excluded_csv}")

    return fig, plot_df


def create_variance_ratio_map(comp_dir, output_dir=None, pattern='*_WO-*_VS_*.PKL', min_num_points=30, min_h_std=0.0,
                              figsize=(14, 10), dpi=300, cmap='coolwarm'):
    if output_dir is None:
        output_dir = os.path.join(comp_dir, "MAPS")

    compiled_data, sum_components, compare_with, solution = load_station_results(comp_dir, pattern=pattern)

    if not compiled_data:
        print("No data available for visualization")
        return None, None

    mapping_stats, filtered_out_stations = extract_statistics_for_mapping(compiled_data, min_num_points, min_h_std)

    if not mapping_stats:
        print("No stations meet the criteria for visualization")
        return None, None

    try:
        station_coords = pd.read_pickle('EXT/PROCESSINS_SUPPLEMENTS/ALL_STATIONS_LATLON.PKL')
    except Exception as e:
        print(f"Error loading station coordinates: {str(e)}")
        return None, None

    plot_data = []
    for sta, stats in mapping_stats.items():
        if sta in station_coords.index:
            lat = station_coords.loc[sta, 'Latitude']
            lon = station_coords.loc[sta, 'Longitude']

            std_original = stats['std_original']
            std_comparison = stats['std_comparison']
            variance_ratio = (std_comparison ** 2 / std_original ** 2) * 100 if std_original > 0 else 0

            if not np.isnan(lat) and not np.isnan(lon):
                plot_data.append({
                    'station': sta,
                    'lat': lat,
                    'lon': lon,
                    'variance_explained': stats['variance_explained'],
                    'variance_ratio': variance_ratio,
                    'std_original': std_original,
                    'std_comparison': std_comparison,
                    'num_points': stats['num_points']
                })

    excluded_data = []
    for sta, stats in filtered_out_stations.items():
        if sta in station_coords.index:
            lat = station_coords.loc[sta, 'Latitude']
            lon = station_coords.loc[sta, 'Longitude']

            if not np.isnan(lat) and not np.isnan(lon):
                excluded_data.append({
                    'station': sta,
                    'lat': lat,
                    'lon': lon,
                    'reason': stats['reason']
                })

    if not plot_data:
        return None, None

    plot_df = pd.DataFrame(plot_data)
    excluded_df = pd.DataFrame(excluded_data) if excluded_data else pd.DataFrame()

    excluded_by_points = sum(1 for _, row in excluded_df.iterrows() if row['reason'] == 'not_enough_points')
    excluded_by_h_std = sum(1 for _, row in excluded_df.iterrows() if row['reason'] == 'h_std_too_low')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ax.set_global()

    if not excluded_df.empty:
        ax.scatter(
            excluded_df['lon'], excluded_df['lat'],
            transform=ccrs.PlateCarree(),
            color='grey',
            s=15,
            alpha=0.5,
            edgecolor=None,
            zorder=5,
            label='Excluded stations'
        )

    scatter = ax.scatter(
        plot_df['lon'], plot_df['lat'],
        transform=ccrs.PlateCarree(),
        c=plot_df['variance_ratio'],
        cmap=cmap,
        vmin=0,
        vmax=100,
        s=50,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
        zorder=10,
        label='Included stations'
    )

    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label(f'Ratio of {compare_with} Variance to Original Variance (%)')

    top_n = 15
    top_stations = plot_df.nlargest(top_n, 'variance_ratio')

    for _, row in top_stations.iterrows():
        ax.text(row['lon'], row['lat'] + 2, row['station'],
                transform=ccrs.PlateCarree(),
                fontsize=8, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', linewidth=0))

    title = f'Variance Ratio Analysis for {solution}: {compare_with} vs Original (without {sum_components})'
    subtitle = f'Stations with at least {min_num_points} data points and H std ≥ {min_h_std} mm'
    ax.set_title(f'{title}\n{subtitle}', fontsize=14)

    stats_text = (
        f"Total stations: {len(plot_df)}\n"
        f"Excluded stations: {len(excluded_df)} (Points: {excluded_by_points}, H std: {excluded_by_h_std})\n"
        f"Mean variance ratio: {plot_df['variance_ratio'].mean():.2f}%\n"
        f"Median variance ratio: {plot_df['variance_ratio'].median():.2f}%\n"
        f"Range: {plot_df['variance_ratio'].min():.2f}% - {plot_df['variance_ratio'].max():.2f}%"
    )

    plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir,
                                   f'{solution}_variance_ratio_map_WO-{sum_components}_VS_{compare_with}.png')
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    return fig, plot_df


def create_correlation_map(comp_dir, value_to_plot, output_dir=None, pattern='*_WO-*_VS_*.PKL', min_num_points=30, min_h_std=0.0,
                           figsize=(14, 10), dpi=300, cmap='plasma'):
    if output_dir is None:
        output_dir = os.path.join(comp_dir, "MAPS")

    compiled_data, sum_components, compare_with, solution = load_station_results(comp_dir, pattern=pattern)

    if not compiled_data:
        print("No data available for visualization")
        return None, None

    mapping_stats, filtered_out_stations = extract_statistics_for_mapping(compiled_data, min_num_points, min_h_std)

    if not mapping_stats:
        print("No stations meet the criteria for visualization")
        return None, None

    try:
        station_coords = pd.read_pickle('EXT/PROCESSINS_SUPPLEMENTS/ALL_STATIONS_LATLON.PKL')
    except Exception as e:
        print(f"Error loading station coordinates: {str(e)}")
        return None, None

    plot_data = []
    for sta, stats in mapping_stats.items():
        if sta in station_coords.index:
            lat = station_coords.loc[sta, 'Latitude']
            lon = station_coords.loc[sta, 'Longitude']

            if not np.isnan(lat) and not np.isnan(lon):
                plot_data.append({
                    'station': sta,
                    'lat': lat,
                    'lon': lon,
                    value_to_plot: stats[value_to_plot],
                    'variance_explained': stats['variance_explained'],
                    'num_points': stats['num_points']
                })

    excluded_data = []
    for sta, stats in filtered_out_stations.items():
        if sta in station_coords.index:
            lat = station_coords.loc[sta, 'Latitude']
            lon = station_coords.loc[sta, 'Longitude']

            if not np.isnan(lat) and not np.isnan(lon):
                excluded_data.append({
                    'station': sta,
                    'lat': lat,
                    'lon': lon,
                    'reason': stats['reason']
                })

    if not plot_data:
        print("No stations with valid coordinates found")
        return None, None

    plot_df = pd.DataFrame(plot_data)
    excluded_df = pd.DataFrame(excluded_data) if excluded_data else pd.DataFrame()

    excluded_by_points = sum(1 for _, row in excluded_df.iterrows() if row['reason'] == 'not_enough_points')
    excluded_by_h_std = sum(1 for _, row in excluded_df.iterrows() if row['reason'] == 'h_std_too_low')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.MaxNLocator(nbins=6)
    gl.ylocator = mticker.MaxNLocator(nbins=6)

    ax.set_global()

    if not excluded_df.empty:
        ax.scatter(
            excluded_df['lon'], excluded_df['lat'],
            transform=ccrs.PlateCarree(),
            color='grey',
            s=15,
            alpha=0.5,
            edgecolor=None,
            zorder=5,
            label='Excluded stations'
        )

    scatter = ax.scatter(
        plot_df['lon'], plot_df['lat'],
        transform=ccrs.PlateCarree(),
        c=plot_df[value_to_plot],
        cmap=cmap,
        vmin=-1,
        vmax=1,
        s=50,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
        zorder=10,
        label='Included stations'
    )

    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
    cbar.set_label(f'Correlation between Original Data and {compare_with}')

    top_n = 15
    top_stations = plot_df.nlargest(top_n, value_to_plot)

    for _, row in top_stations.iterrows():
        ax.text(row['lon'], row['lat'] + 2, row['station'],
                transform=ccrs.PlateCarree(),
                fontsize=8, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', linewidth=0))

    title = f'Correlation Analysis for {solution}: {compare_with} vs Original (without {sum_components})'
    subtitle = f'Stations with at least {min_num_points} data points and H std ≥ {min_h_std} mm'
    ax.set_title(f'{title}\n{subtitle}', fontsize=14)

    stats_text = (
        f"Total stations: {len(plot_df)}\n"
        f"Excluded stations: {len(excluded_df)} (Points: {excluded_by_points}, H std: {excluded_by_h_std})\n"
        f"Mean correlation: {plot_df[value_to_plot].mean():.2f}\n"
        f"Median correlation: {plot_df[value_to_plot].median():.2f}\n"
        f"Range: {plot_df[value_to_plot].min():.2f} - {plot_df[value_to_plot].max():.2f}"
    )

    plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir,
                                   f'{solution}_{value_to_plot}_map_WO-{sum_components}_VS_{compare_with}.png')
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

        csv_file = os.path.join(output_dir,
                                f'{solution}_{value_to_plot}_data_WO-{sum_components}_VS_{compare_with}.csv')
        plot_df.to_csv(csv_file, index=False)
        print(f"Data saved to {csv_file}")

    return fig, plot_df


def create_top_stations_bar_plot(plot_df, metric='variance_explained', top_percent=10,
                                 figsize=(6, 8), output_dir=None, solution=None, sum_components=None,
                                 compare_with=None, dpi=300):
    if plot_df is None or len(plot_df) == 0:
        print("No data available for bar plot")
        return None

    if metric not in plot_df.columns:
        print(f"Metric '{metric}' not found in data")
        return None

    num_stations = max(1, int(len(plot_df) * top_percent / 100))
    top_stations = plot_df.nlargest(num_stations, metric)
    top_stations = top_stations.sort_values(by=metric)

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(top_stations['station'], top_stations[metric],
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(0.5, bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}', ha='left', va='center', fontsize=8)

    metric_label = ' '.join(word.capitalize() for word in metric.split('_'))
    ax.set_xlabel(f'{metric_label} (%)')
    ax.set_ylabel('Station')

    title = f'Top {top_percent}% Stations \nby {metric_label}'
    if solution and compare_with:
        title += f' for {solution}'
        if sum_components:
            title += f' (without {sum_components}, compared with {compare_with})'
    ax.set_title(title)

    plt.tight_layout()

    if output_dir and solution and compare_with:
        os.makedirs(output_dir, exist_ok=True)
        metric_str = '_'.join(metric.split('_'))
        output_file = os.path.join(output_dir,
                                   f'{solution}_top_{top_percent}percent_{metric_str}_WO-{sum_components}_VS_{compare_with}.png')
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Bar plot saved to {output_file}")

    return fig

# Example usage
if __name__ == "__main__":
    solution = 'IGS1R03SNX'
    sampling = '01D'
    reduction = 'None'
    vs = 'A'

    # Example usage with default directory
    comp_dir = f'OUTPUT/SNX_LOAD_COMPARISONS/{solution}_{sampling}/PKL'
    output_dir = os.path.join(os.path.dirname(comp_dir), "MAPS")
    min_num_points = 1000
    min_h_std = 1.5  # 1.5 mm minimum standard deviation for H component
    pattern = f'*_WO-{reduction}_VS_{vs}*.PKL'

    print(f"Using parameters: min_num_points={min_num_points}, min_h_std={min_h_std}")
    print(f"Pattern: {pattern}")
    print(f"Output directory: {output_dir}")

    # Create the variance explained map
    fig1, plot_df1 = visualize_variance_explained_map(comp_dir, output_dir, pattern, min_num_points, min_h_std, cmap='Greens')

    # Create the variance ratio map
    fig2, plot_df2 = create_variance_ratio_map(comp_dir, output_dir, pattern, min_num_points, min_h_std)

    # Create the correlation map
    fig3, plot_df3 = create_correlation_map(comp_dir, 'correlation', output_dir, pattern, min_num_points, min_h_std, cmap='coolwarm')
    fig3a, plot_df3a = create_correlation_map(comp_dir, 'kge2012', output_dir, pattern, min_num_points, min_h_std, cmap='coolwarm')

    # Determine sum_components, compare_with, solution from the data
    compiled_data, sum_components, compare_with, solution = load_station_results(comp_dir, pattern)
    #
    # # Create bar plot for top stations with highest variance explained
    fig4 = create_top_stations_bar_plot(plot_df1, metric='variance_explained', top_percent=10,
                                        output_dir=output_dir, solution=solution,
                                        sum_components=sum_components, compare_with=compare_with)

    # # Create bar plot for top stations with highest correlation
    fig5 = create_top_stations_bar_plot(plot_df3, metric='correlation', top_percent=10,
                                        output_dir=output_dir, solution=solution,
                                        sum_components=sum_components, compare_with=compare_with)

    # Show the plots
    plt.close('all')