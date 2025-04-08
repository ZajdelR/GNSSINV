# Geodetic Station Displacement Analysis Toolkit

This repository contains a collection of Python scripts for analyzing global geodetic measurements, including station displacements from various techniques (GNSS, SLR, VLBI) based on SINEX files, and their relationship to various geophysical loading phenomena. The toolkit enables the processing of SINEX files, computation of displacements, comparison with loading models, visualization of results through global maps and time series analysis, and global inversion of station displacements to retrieve low-degree gravity field coefficients.

## Project Structure

The project consists of several Python scripts for different processing steps:

### Data Acquisition and Preprocessing

- `01_read_apr_est_from_snx.py`: Processes SINEX files and extracts station coordinates, transforming from STAX/STAY/STAZ format to a unified table with X, Y, Z components.
- `01_read_esm_loadings_files.py`: Processes GFZ loading files and saves them as pickle files with consistent daily sampling.
- `01_read_ITRF_res_files.py`: Processes ITRF residual files, converting them to displacement time series organized by both station and time.

### Displacement Calculation

- `02_displacements_from_snx_est_apr.py`: Computes displacements by comparing estimated and a priori coordinates, transforms from ECEF to topocentric coordinates, and optionally adds periodic signals.
- `02_displacements_from_esmgfz_nc_grids.py`: Converts NetCDF grid files to daily displacement dataframes with specific grid resolution.

### Loading Model Integration

- `03_displacements_from_esmgfz_point.py`: Processes GFZ loading model data and organizes them by time for comparison with GNSS displacements.

### Displacement Analysis

- `04_1_analysis_compare_snx_displacements_with_load.py`: Compares GNSS displacements with loading model predictions, calculates statistics, and performs spectral analysis.
- `04_2_global_map_analysis_consistency.py`: Creates global maps showing the spatial patterns of agreement between GNSS and loading models.

### Gravity Field Inversion

- `05_displacement_inversion.py`: Converts displacement data to spherical harmonic coefficients and computes gravity field variations.

### Coefficient Analysis

- `06_analysis_gfc_c20c30.py`: Analyzes temporal variations in key coefficients (C20, C30) and compares with reference solutions.
- `06_analysis_gfc_degree1.py`: Analyzes degree-1 coefficients (geocenter motion) from different solutions.

## Methodology

### Data Processing Pipeline

The general workflow involves:

1. **SINEX File Processing**:
   - Parse SINEX coordinate files containing station position estimates from different geodetic techniques (GNSS, SLR, VLBI)
   - Transform coordinates from STAX/STAY/STAZ format to a unified table
   - Store results as pickle files for efficient access

2. **Displacement Calculation**:
   - Compute displacements by comparing estimated and a priori coordinates
   - Transform from ECEF (Earth-Centered, Earth-Fixed) to local topocentric coordinates (North, East, Up)
   - Optionally add periodic signals based on annual and semi-annual coefficients
   - Organize results by station and by time

3. **Loading Data Integration**:
   - Process GFZ loading model data (atmospheric, hydrological, oceanic, surface water)
   - Format loading data to match station displacement data structure for comparison

4. **Displacement Analysis**:
   - Compare station displacements with loading model predictions
   - Calculate statistics including correlation, variance explained, RMS
   - Perform Lomb-Scargle spectral analysis to identify periodic signals
   - Generate visualizations of time series and periodograms

5. **Global Analysis Maps**:
   - Create global maps showing spatial patterns of agreement between observed displacements and loading models
   - Visualize variance explained, correlation, and other metrics across the global station network
   - Filter stations based on criteria such as time series length and signal strength

6. **Gravity Field Inversion**:
   - Convert station displacement data to spherical harmonic coefficients through global inversion
   - Compute low-degree gravity field coefficients from observed displacements
   - Account for different reference frames (CM, CF) in the inversion process
   - Validate the solutions by reconstructing the original displacements

7. **Coefficient Analysis**:
   - Compare spherical harmonic coefficients from different solutions
   - Analyze temporal variations in key coefficients (C20, C30, degree-1)
   - Compare with reference solutions (e.g., SLR-derived coefficients)

### Key Analysis Methods

#### Comparing Station Displacements with Loading Models

The toolkit quantifies the agreement between observed station displacements and loading models using:

- **Variance Explained**: Percentage of displacement signal variance that can be explained by loading models
- **Correlation Coefficient**: Pearson correlation between observed and model time series
- **RMS Difference**: Root mean square of residuals between observed and model time series
- **Kling-Gupta Efficiency (KGE)**: A composite measure that accounts for correlation, bias, and variability ratio

The analysis is performed by:
1. Removing selected loading effects from observed station displacements
2. Comparing the residual signal with another loading component
3. Computing statistics and visualizing time series and spectral properties

#### Global Inversion for Gravity Field Coefficients

The toolkit can perform global inversion of station displacements to retrieve low-degree gravity field coefficients:

- **Spherical Harmonic Decomposition**: Converting station displacements to spherical harmonic coefficients
- **Love Number Theory**: Using elastic Earth models to relate surface displacements to mass redistribution
- **Reference Frame Handling**: Accounting for different reference frames (CM, CF) in the inversion process
- **Regularization Options**: Including Tikhonov regularization for stabilizing the inversion when needed

### Validation Approach

The validity of solutions is assessed through:

- **Reconstructed Displacements**: Converting computed spherical harmonics back to station displacements and comparing with the original observations
- **Variance Explanation Analysis**: Quantifying how much of the original signal variance is captured
- **Comparison with Reference Solutions**: Comparing derived coefficients with independent solutions (e.g., TN-14 for C20/C30)

## Data Requirements

The scripts expect the following directory structure:

```
├── DATA/
│   ├── DISPLACEMENTS/
│   │   ├── {solution}_{sampling}/
│   │   │   ├── CODE/         # Displacement data organized by station code
│   │   │   └── TIME/         # Displacement data organized by date
│   ├── SNX_ORIGINAL/         # Original SINEX files
│   └── SNX_OUT_PKL/          # Processed SINEX data as pickle files
├── EXT/
│   ├── ESMGFZLOADING/
│   │   └── CODE/             # Loading model data by station
│   ├── ITRF_PERIODIC_MODEL/  # Periodic signal models
│   ├── LLNs/                 # Love number files
│   ├── PROCESSINS_SUPPLEMENTS/ # Station metadata
│   └── GCC/                  # Geocenter motion data
├── OUTPUT/                   # Output directory for results
│   └── {solution}/
│       └── TIME/
│           └── {date}/
│               ├── gravity_coeffs_{identifier}.txt
│               └── validation/
└── LOGS/                     # Log files
```

### Data Sources

- **SINEX Files**: 
  - GNSS: Analysis Center contributions to repro3 are publicly available at https://cddis.nasa.gov/archive/gnss/products/wwww/repro3/, where *wwww* stands for the 4-character GPS week number
  - SLR and VLBI: Available from ITRF combination centers (can be processed with the same tools)

- **Loading Model Data**: 
  - ESMGFZ loading deformation models are available from http://rz-vm115.gfz-potsdam.de:8080/repository
  - GGFC loading deformation models are available from http://loading.u-strasbg.fr/ITRF2020
  - For processing GFZ loading grid data, external software is required and available at: http://rz-vm115.gfz.de:8080/repository/entry/show?entryid=97d15ffe-3b5d-49dc-ba6c-3011851af1de

## Usage

The typical workflow involves running the scripts in numerical order:

1. Process SINEX files to extract coordinates:
   ```
   python 01_read_apr_est_from_snx.py
   ```

2. Calculate displacements from coordinates:
   ```
   python 02_displacements_from_est_apr_snx.py
   ```

3. Process loading model data:
   ```
   python 01_read_esm_loadings_files.py
   python 03_displacements_from_esmgfz_point.py
   ```

4. Compare displacements with loading models:
   ```
   python 04_1_analysis_compare_snx_displacements_with_load.py
   ```

5. Generate global maps and statistics:
   ```
   python 04_2_global_map_analysis_consistency.py
   ```

6. Perform gravity field inversions:
   ```
   python 05_displacement_inversion.py --solution IGS1R03SNX_01D --max-degree 7 --start-date 20180101 --end-date 20180102
   ```

7. Analyze spherical harmonic coefficients:
   ```
   python 06_analysis_gfc_c20c30.py
   python 06_analysis_gfc_degree1.py
   ```

Each script includes command-line arguments to customize processing parameters.

## Dependencies

The scripts require the following Python libraries:

- pandas
- numpy
- matplotlib
- cartopy
- scipy
- xarray
- astropy
- geodezyx (a specialized geodetic library)
- concurrent.futures (for parallel processing)
- pyyaml (for configuration files)
- statsmodels
- tqdm (for progress bars)

## References

For a more detailed explanation of the mathematical background, please refer to:

- Blewitt, G. (2003). Self-consistency in reference frames, geocenter definition, and surface loading of the solid earth. *Journal of Geophysical Research: Solid Earth*, *108*(B2). https://doi.org/10.1029/2002JB002082
- Blewitt, G., & Clarke, P. (2003). Inversion of Earth's changing shape to weigh sea level in static equilibrium with surface mass redistribution. *Journal of Geophysical Research: Solid Earth*, *108*(B6). https://doi.org/10.1029/2002JB002290
- Fritsche, M., Dietrich, R., Rülke, A., Rothacher, M., & Steigenberger, P. (2010). Low-degree earth deformation from reprocessed GPS observations. *GPS Solutions*, *14*(2), 165–175. https://doi.org/10.1007/s10291-009-0130-7
- Kusche, J., & Schrama, E. J. O. (2005). Surface mass redistribution inversion from global GPS deformation and gravity recovery and climate experiment (GRACE) gravity data. *Journal of Geophysical Research: Solid Earth*, *110*(B9). https://doi.org/10.1029/2004JB003556
- Meyrath, T., van Dam, T., Collilieux, X., & Rebischung, P. (2017). Seasonal low-degree changes in terrestrial water mass load from global GNSS measurements. *Journal of Geodesy*, *91*(11), 1329–1350. https://doi.org/10.1007/s00190-017-1028-8
- Nowak, A., Zajdel, R., Gałdyn, F., & Sośnica, K. (2024). Low-degree gravity field coefficients based on inverse GNSS method: Insights into hydrological and ice mass change studies. *GPS Solutions*, *29*(1), 5. https://doi.org/10.1007/s10291-024-01760-1
- Wang, H., Xiang, L., Jia, L., Jiang, L., Wang, Z., Hu, B., & Gao, P. (2012). Load Love numbers and Green's functions for elastic Earth models PREM, iasp91, ak135, and modified models with refined crustal structure from Crust 2.0. *Computers & Geosciences*, *49*, 190–199. https://doi.org/10.1016/j.cageo.2012.06.022

## Contributors

This toolkit was developed by Radoslaw Zajdel

## License

This software is released under the MIT License.