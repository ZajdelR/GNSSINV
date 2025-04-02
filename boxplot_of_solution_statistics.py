import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

import matplotlib
matplotlib.use('TkAgg')

def create_boxplot_comparison(file_pattern, inp_dir, column_name, output_filename=None, solution=''):
    """
    Creates a boxplot comparing the specified column across multiple CSV files.

    Parameters:
    -----------
    file_pattern : str
        Pattern to match CSV files (e.g., "IGS1R03SNX_variance_explained_data_*.csv")
    inp_dir : str
        Input directory where the CSV files are located
    column_name : str
        Name of the column to compare (e.g., "variance_explained")
    output_filename : str, optional
        Name of the output file. If None, the plot will be displayed instead of saved.
    """
    # Find all files matching the pattern
    files = glob.glob(os.path.join(inp_dir, file_pattern))

    if not files:
        print(f"No files found matching the pattern: {file_pattern}")
        return

    print(f"Found {len(files)} files matching the pattern.")

    # Create a dictionary to store data from each file
    data_dict = {}

    # Read each file and extract the column of interest
    for file_path in files:
        try:
            # Extract a more readable name from the file path
            file_name = os.path.basename(file_path)
            # Remove common prefix and suffix to get a cleaner label
            label = file_name.replace("IGS1R03SNX_variance_explained_data_", "").replace("_VS_H.csv", "")

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check if the column exists
            if column_name not in df.columns:
                print(
                    f"Warning: Column '{column_name}' not found in {file_name}. Available columns: {', '.join(df.columns)}")
                continue

            # Store the column data
            data_dict[label] = df[column_name]

            # Print basic statistics
            print(f"\nStatistics for {label}:")
            print(f"  Count: {len(df[column_name])}")
            print(f"  Mean: {df[column_name].mean():.4f}")
            print(f"  Median: {df[column_name].median():.4f}")
            print(f"  Min: {df[column_name].min():.4f}")
            print(f"  Max: {df[column_name].max():.4f}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not data_dict:
        print("No data was extracted from the files.")
        return

    # Convert the dictionary to a DataFrame for easier plotting
    plot_data = pd.DataFrame(data_dict)

    # Set the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # Create the boxplot
    ax = sns.boxplot(data=plot_data, orient='v', whis=[0,100])

    # # Add individual data points - Fixed line that was causing the error
    # for i, label in enumerate(data_dict.keys()):
    #     sns.stripplot(x=[i] * len(data_dict[label]), y=data_dict[label], color='black', size=3, alpha=0.3, jitter=True, ax=ax)

    # Add title and labels
    plt.title(f'{column_name} {solution}', fontsize=16)
    plt.ylabel(column_name, fontsize=14)
    plt.xlabel('Dataset', fontsize=14)
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Save or display the plot
    if output_filename:
        plt.savefig(os.path.join(inp_dir, output_filename), dpi=300, bbox_inches='tight')
        print(f"Plot saved as {os.path.join(inp_dir, output_filename)}")
    else:
        plt.show()

    # Generate summary table
    summary = pd.DataFrame({
        'Dataset': list(data_dict.keys()),
        'Count': [len(data_dict[k]) for k in data_dict],
        'Mean': [data_dict[k].mean() for k in data_dict],
        'Median': [data_dict[k].median() for k in data_dict],
        'Min': [data_dict[k].min() for k in data_dict],
        'Max': [data_dict[k].max() for k in data_dict],
        'Std Dev': [data_dict[k].std() for k in data_dict]
    })

    print("\nSummary Statistics:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Optionally save the summary statistics to a CSV file
    if output_filename:
        summary_filename = os.path.join(inp_dir, f"{column_name}_summary.csv")
        summary.to_csv(summary_filename, index=False)
        print(f"Summary statistics saved to {summary_filename}")

    return summary


if __name__ == "__main__":
    # Example usage
    solution = 'IGS1R03SNX'
    sampling = '01D'
    inp_dir = rf'INPUT_CRD/{solution}_{sampling}/COMP/MAPS'
    file_pattern = f"{solution}_variance_explained_data_*.csv"
    column_name = "variance_explained"
    output_filename = f"{column_name}_boxplot.png"

    # Create the boxplot
    summary = create_boxplot_comparison(file_pattern, inp_dir,
                                        column_name,
                                        output_filename,
                                        solution=f'{solution}_{sampling}')