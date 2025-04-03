# GNSS Displacement Analysis Toolkit

This repository contains a collection of Python scripts for analyzing global geodetic measurements, particularly GNSS station displacements, and their relationship to various geophysical loading phenomena. The toolkit enables the processing of SINEX files, computation of displacements, comparison with loading models, and visualization of results through global maps and time series analysis.

## Methodology

### Data Processing Pipeline

Our methodology comprises several sequential processing steps:

1. **SINEX File Processing** (`01_read_snx_crd_files.py`):
   - Parse SINEX coordinate files containing station position estimates
   - Transform coordinates from STAX/STAY/STAZ format to a unified table
   - Store results as pickle files for efficient access

2. **Displacement Calculation** (`02_displacements_from_est_apr_snx.py`):
   - Compute displacements by comparing estimated and a priori coordinates
   - Add periodic signals based on annual and semi-annual coefficients
   - Transform from ECEF (Earth-Centered, Earth-Fixed) to local topocentric coordinates (North, East, Up)
   - Organize results by station and by time

3. **Loading Data Integration** (`03_displacements_from_esmgfz.py`):
   - Process GFZ loading model data (atmospheric, hydrological, oceanic, surface water)
   - Format loading data to match GNSS displacement data structure for comparison

4. **Displacement Analysis** (`04_1_analysis_compare_snx_displacements_with_load.py`):
   - Compare GNSS displacements with loading model predictions
   - Calculate statistics including correlation, variance explained, RMS
   - Perform Lomb-Scargle spectral analysis to identify periodic signals
   - Generate visualizations of time series and periodograms

5. **Global Analysis Maps** (`04_2_global_map_analysis_consistency.py`):
   - Create global maps showing spatial patterns of agreement between GNSS and loading models
   - Visualize variance explained, correlation, and other metrics across the global station network
   - Filter stations based on criteria such as time series length and signal strength

6. **Gravity Field Inversion** (`04_displacement_inversion.py`):
   - Convert displacement data to spherical harmonic coefficients
   - Compute gravity field variations from displacement observations
   - Validate the solutions by reconstructing the original displacements

7. **Coefficient Analysis** (`05_analysis_gfc_c20c30.py`, `05_analysis_gfc_degree1.py`):
   - Compare spherical harmonic coefficients from different solutions
   - Analyze temporal variations in key coefficients (C20, C30, degree-1)
   - Compare with reference solutions (e.g., SLR-derived coefficients)

### Key Analysis Methods

#### Comparing GNSS Displacements with Loading Models

We quantify the agreement between GNSS observations and loading models using:

- **Variance Explained**: Percentage of GNSS signal variance that can be explained by loading models
- **Correlation Coefficient**: Pearson correlation between GNSS and loading time series
- **RMS Difference**: Root mean square of residuals between GNSS and loading time series
- **Kling-Gupta Efficiency (KGE)**: A composite measure that accounts for correlation, bias, and variability ratio

The analysis is performed by:
1. Removing selected loading effects from GNSS observations
2. Comparing the residual signal with another loading component
3. Computing statistics and visualizing time series and spectral properties

### Validation Approach

The validity of our solutions is assessed through:

- **Reconstructed Displacements**: Converting computed spherical harmonics back to station displacements and comparing with the original observations
- **Variance Explanation Analysis**: Quantifying how much of the original signal variance is captured
- **Comparison with Reference Solutions**: Comparing derived coefficients with independent solutions (e.g., TN-14 for C20/C30)

## Data Requirements

The scripts expect the following data structure:

- SINEX files with station coordinates
- Loading model data (atmospheric, hydrological, oceanic, surface water)
- Station metadata including coordinates and availability information
- Love number files for displacement-to-gravity conversion

### Data Sources

- **SINEX Files**: All Analysis Center contributions to repro3 are publicly available at https://cddis.nasa.gov/archive/gnss/products/wwww/repro3/, where *wwww* stands for the 4-character GPS week number.

- **Loading Model Data**: 
  - ESMGFZ loading deformation models are available from http://rz-vm115.gfz-potsdam.de:8080/repository
  - GGFC loading deformation models are available from http://loading.u-strasbg.fr/ITRF2020

## Usage

The typical workflow involves running the scripts in numerical order:

1. Process SINEX files to extract coordinates
2. Calculate displacements from coordinates
3. Compare displacements with loading models
4. Generate global maps and statistics
5. Perform gravity field inversions
6. Analyze spherical harmonic coefficients

Each script includes command-line arguments to customize processing parameters.

## Dependencies

The scripts require the following Python libraries:

- pandas
- numpy
- matplotlib
- cartopy
- scipy
- astropy
- statsmodels
- geodezyx (a specialized geodetic library)

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