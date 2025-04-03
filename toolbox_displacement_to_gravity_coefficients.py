import numpy as np
from pyshtools import SHCoeffs
from scipy.special import sph_harm, lpmv
import scipy.special
import os
from toolbox_gravity_validation import validate_coefficient_solution

def add_translation_rotation_parameters(A, n_sites, lats, lons):
    """
    Add translation and rotation parameters to the design matrix.

    Parameters:
    -----------
    A : ndarray
        Design matrix
    n_sites : int
        Number of sites
    lats : ndarray
        Latitudes of sites (degrees)
    lons : ndarray
        Longitudes of sites (degrees)

    Returns:
    --------
    A_extended : ndarray
        Extended design matrix with 6 additional columns
    """
    # Expand A matrix to add 6 columns: 3 for translation, 3 for rotation
    A_extended = np.zeros((A.shape[0], A.shape[1] + 6))
    A_extended[:, :A.shape[1]] = A

    for i in range(n_sites):
        # Convert to radians
        phi = np.radians(lats[i])  # Latitude in radians
        lam = np.radians(lons[i])  # Longitude in radians

        # Create unit vectors in local north, east, and up directions
        e_n = np.array([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)])
        e_e = np.array([-np.sin(lam), np.cos(lam), 0])
        e_u = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])

        # North component
        A_extended[i, A.shape[1]:A.shape[1] + 3] = e_n  # Translation
        A_extended[i, A.shape[1] + 3] = 0  # rx
        A_extended[i, A.shape[1] + 4] = -e_u[2]  # ry - rotates around y-axis
        A_extended[i, A.shape[1] + 5] = e_u[1]  # rz - rotates around z-axis

        # East component
        A_extended[i + n_sites, A.shape[1]:A.shape[1] + 3] = e_e  # Translation
        A_extended[i + n_sites, A.shape[1] + 3] = e_u[2]  # rx - rotates around x-axis
        A_extended[i + n_sites, A.shape[1] + 4] = 0  # ry
        A_extended[i + n_sites, A.shape[1] + 5] = -e_u[0]  # rz - rotates around z-axis

        # Up component
        A_extended[i + 2 * n_sites, A.shape[1]:A.shape[1] + 3] = e_u  # Translation
        A_extended[i + 2 * n_sites, A.shape[1] + 3] = -e_e[2]  # rx - rotates around x-axis
        A_extended[i + 2 * n_sites, A.shape[1] + 4] = e_e[1]  # ry - rotates around y-axis
        A_extended[i + 2 * n_sites, A.shape[1] + 5] = 0  # rz

    return A_extended


def calculate_load_coefficients(displacements, max_degree=6, love_numbers_file=None, calculate_errors=False,
                                reference_frame="CF", add_helmert=False, regularize=True,
                                damping_factor=5e-3, degree_dependent=True, auto_dumping=False,
                                solving_system_method ='svd'):
    """
    Calculate load coefficients from site displacements using unified least squares adjustment.

    Returns load coefficients representing surface mass density in kg/m².
    """
    # Constants
    R_E = 6371000.0  # Earth radius (m)
    M_E = 5.972e24  # Earth mass (kg)
    rho_w = 1025.0  # Density of seawater (kg/m³)
    rho_E = 5517.0  # Average density of Earth (kg/m³)
    g = 9.80665  # Gravitational acceleration at Earth's surface (m/s²)

    # Extract site data
    u_values = np.array(displacements['vertical'])  # Vertical displacement (Up)
    e_values = np.array(displacements['east'])  # East displacement
    n_values = np.array(displacements['north'])  # North displacement
    lats = np.array(displacements['lat'])  # Latitudes (φ)
    lons = np.array(displacements['lon'])  # Longitudes (λ)

    # Load Love numbers (in CE frame)
    love_numbers = load_love_numbers(love_numbers_file)

    # Transform degree-1 load Love numbers based on reference frame
    transformed_love_numbers = transform_love_numbers(love_numbers, reference_frame)

    print(f"Using reference frame: {reference_frame}")
    print(f"Degree-1 load Love numbers: h'₁={transformed_love_numbers['h_n'][1]:.7f}, "
          f"l'₁={transformed_love_numbers['l_n'][1]:.7f}, "
          f"k'₁={transformed_love_numbers['k_n'][1]:.7f}")

    # Number of sites
    n_sites = len(lats)

    # List of all (n, m) pairs for the coefficients
    nm_pairs = []
    for n in range(1, max_degree + 1):
        for m in range(0, n + 1):
            nm_pairs.append((n, m))

    # Number of coefficients
    n_coeff_pairs = len(nm_pairs)
    n_coeffs = 2 * n_coeff_pairs - (max_degree)  # Subtract max_degree because Sn0 = 0 for all n

    print(f"Setting up least squares problem with {n_sites * 3} equations and {n_coeffs} unknowns...")

    # Initialize the design matrix and observation vector
    A = np.zeros((n_sites * 3, n_coeffs))
    y = np.zeros(n_sites * 3)

    # Fill observation vector
    y[:n_sites] = n_values  # North component
    y[n_sites:2 * n_sites] = e_values  # East component
    y[2 * n_sites:] = u_values  # Up/Vertical component

    # Map coefficient indices
    coeff_idx = {}
    idx = 0
    for n, m in nm_pairs:
        # C coefficient
        coeff_idx[(n, m, 'C')] = idx
        idx += 1

        # S coefficient (only for m > 0)
        if m > 0:
            coeff_idx[(n, m, 'S')] = idx
            idx += 1

    print("Building design matrix based on the provided equations...")
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if i % 100 == 0 or i == n_sites - 1:
            print(f"  Processing site {i + 1}/{n_sites}...")

        # Convert to appropriate angles
        phi_i = np.radians(lat)
        lam_i = np.radians(lon)
        cos_phi = np.cos(phi_i)

        for n, m in nm_pairs:
            # Get Love numbers (already transformed for the chosen reference frame)
            h_n = transformed_love_numbers['h_n'][n]  # Vertical Love number
            l_n = transformed_love_numbers['l_n'][n]  # Horizontal Love number

            # Compute associated Legendre function
            # z = np.cos(phi_i)

            z = np.clip(np.cos(phi_i), -1 + 1e-10, 1 - 1e-10)

            P, dP = scipy.special.lpmn(m, n, z)
            P_nm = P[m, n]
            dP_nm_dz = dP[m, n]

            if 0:
                # Convert to fully normalized Legendre functions (4π normalization)
                if m == 0:
                    norm_factor = np.sqrt((2 * n + 1))
                else:
                    norm_factor = np.sqrt(2 * (2 * n + 1) *
                                          scipy.special.factorial(n - m) /
                                          scipy.special.factorial(n + m))

                # Apply normalization
                P_nm_norm = P_nm * norm_factor
                dP_nm_dz_norm = dP_nm_dz * norm_factor
            if 1:
                # Convert to fully normalized Legendre functions (4π normalization)
                if m == 0:
                    norm_factor = np.sqrt((2 * n + 1))
                else:
                    # More stable calculation using log factorials
                    log_norm = 0.5 * (np.log(2) + np.log(2 * n + 1) +
                                      scipy.special.gammaln(n - m + 1) -
                                      scipy.special.gammaln(n + m + 1))
                    norm_factor = np.exp(log_norm)

                # Apply normalization
                P_nm_norm = P_nm * norm_factor
                dP_nm_dz_norm = dP_nm_dz * norm_factor

            # Compute derivative with respect to phi
            dP_nm_dphi = -np.sin(phi_i) * dP_nm_dz_norm

            # Compute trigonometric terms
            cos_m_lam = np.cos(m * lam_i)
            sin_m_lam = np.sin(m * lam_i)

            # From Farrell (1972) and Blewitt & Clarke (2003)
            # Factor to convert between load and displacement
            # This follows Wahr et al. (1998) formulation
            factor = (rho_E / 3.0) * ((2.0 * n + 1.0) / (1.0 + transformed_love_numbers['k_n'][n]))
            # const = 4.0 * np.pi * R_E ** 3 / M_E
            # factor = const / (2.0 * n + 1.0)

            # NORTH component coefficients
            # north_factor = (l_n * factor * dP_nm_dphi) / R_E
            north_factor = -factor * l_n * dP_nm_dphi / R_E

            # C coefficient (cosine term)
            c_idx = coeff_idx.get((n, m, 'C'))
            A[i, c_idx] = north_factor * cos_m_lam

            # S coefficient (sine term, only for m > 0)
            if m > 0:
                s_idx = coeff_idx.get((n, m, 'S'))
                A[i, s_idx] = north_factor * sin_m_lam

            # EAST component coefficients
            if abs(cos_phi) > 1e-10:  # Avoid division by zero near poles
                # east_factor = (l_n * factor * (m / cos_phi) * P_nm_norm) / R_E
                east_factor = -factor * l_n * (m / cos_phi) * P_nm_norm / R_E

                # C coefficient (sine term with negative sign because of partial derivative)
                A[i + n_sites, c_idx] = -east_factor * sin_m_lam

                # S coefficient (cosine term, only for m > 0)
                if m > 0:
                    A[i + n_sites, s_idx] = east_factor * cos_m_lam

            # UP/VERTICAL component coefficients
            # up_factor = (h_n * factor * P_nm_norm) / R_E
            up_factor = factor * h_n * P_nm_norm / R_E

            # C coefficient (cosine term)
            A[i + 2 * n_sites, c_idx] = up_factor * cos_m_lam

            # S coefficient (sine term, only for m > 0)
            if m > 0:
                A[i + 2 * n_sites, s_idx] = up_factor * sin_m_lam

    # Add Helmert parameters if requested
    if add_helmert:
        A = add_translation_rotation_parameters(A, n_sites, lats, lons)
        print(f"Added 6 Helmert parameters: 3 translations, 3 rotations")

    # Apply regularization if requested
    if regularize:
        orig_rows = A.shape[0]

        if auto_dumping:
            print("Determining optimal regularization parameter using GCV...")
            damping_factor = find_optimal_damping_factor(
                A, y, coeff_idx, max_degree,
                range_min=1e-10,
                range_max=1e-2,
                degree_dependent=degree_dependent
            )

        print(
            f"Applying Tikhonov regularization (damping_factor={damping_factor}, degree_dependent={degree_dependent})")

        A, y = add_tikhonov_regularization(
            A, y, coeff_idx, max_degree,
            damping_factor=damping_factor,
            degree_dependent=degree_dependent
        )

        print(f"Matrix dimensions before regularization: {orig_rows} x {n_coeffs}")
        print(f"Matrix dimensions after regularization: {A.shape[0]} x {A.shape[1]}")

    if solving_system_method == 'svd':
        # Solve the system using SVD
        u_svd, s_vals, vh_svd = scipy.linalg.svd(A, full_matrices=False)
        cond = s_vals[0] / s_vals[-1]
        print(f"Condition number of design matrix: {cond:.4e}")
        print(f"Smallest singular value: {s_vals[-1]:.4e}")

        if cond > 1e12:
            print("Warning: Design matrix is very ill-conditioned!")

        # Threshold for singular values to be considered "zero"
        threshold = 1e-12 * s_vals[0]

        # Apply truncated SVD solution
        s_inv = np.zeros_like(s_vals)
        s_inv[s_vals > threshold] = 1.0 / s_vals[s_vals > threshold]

        # Compute solution
        x = vh_svd.T @ (s_inv * (u_svd.T @ y))
    elif solving_system_method == 'lsa':
        # Solve the system using Least Squares Adjustment with regularization
        print("Solving system using Least Squares Adjustment...")

        try:
            # Method 1: Use numpy's least squares solver
            x, residuals, rank, s_vals = np.linalg.lstsq(A, y, rcond=None)

            # Calculate condition number
            cond = s_vals[0] / s_vals[-1] if len(s_vals) > 1 else float('inf')
            print(f"Condition number of design matrix: {cond:.4e}")
            print(f"Rank of design matrix: {rank} (out of {A.shape[1]} columns)")

            threshold = 1e-12 * s_vals[0]

            # Check solution quality
            if cond > 1e12:
                print("Warning: Design matrix is very ill-conditioned!")

            # Alternatively, we can use scipy's lsqr solver for large sparse systems
            if A.shape[0] > 10000 or cond > 1e10:
                print("Using LSQR for better numerical stability...")
                x_lsqr = scipy.sparse.linalg.lsqr(A, y, atol=1e-8, btol=1e-8)
                x = x_lsqr[0]  # Extract solution vector
                print(f"LSQR converged with reason: {x_lsqr[1]}, iterations: {x_lsqr[2]}")

        except np.linalg.LinAlgError as e:
            # Fallback to more robust solver if standard LSA fails
            print(f"LSA failed with error: {e}. Falling back to robust solver.")

            # Use scipy's LSMR solver which is more robust to ill-conditioning
            x = scipy.sparse.linalg.lsmr(A, y, atol=1e-8, btol=1e-8)[0]

            # Since we don't have singular values anymore, estimate condition number
            AT_A = A.T @ A
            eigvals = scipy.linalg.eigvalsh(AT_A)
            cond = np.sqrt(eigvals.max() / eigvals.min())
            print(f"Estimated condition number: {cond:.4e}")

    # Calculate residuals
    residuals = y - A @ x
    residual_norm = np.linalg.norm(residuals)
    print(f"Solution complete. Using SVD with truncation at {threshold:.4e}")
    print(f"RMS of residuals: {np.sqrt(residual_norm ** 2 / (3 * n_sites))}")

    # Extract Helmert parameters if added
    if add_helmert:
        translation_params = x[-6:-3]
        rotation_params = x[-3:]
        print(f"Estimated translation parameters: {translation_params}")
        print(f"Estimated rotation parameters: {rotation_params}")
        x = x[:-6]  # Remove Helmert parameters from solution

    # Convert solution vector to coefficient arrays
    coeffs_array = np.zeros((2, max_degree + 1, max_degree + 1))

    # Fill in the coefficient arrays
    for (n, m), idx in [(key[:2], val) for key, val in coeff_idx.items() if key[2] == 'C']:
        coeffs_array[0, n, m] = x[idx]  # Cosine coefficient (C)

    for (n, m), idx in [(key[:2], val) for key, val in coeff_idx.items() if key[2] == 'S']:
        coeffs_array[1, n, m] = x[idx]  # Sine coefficient (S)

    # Create SHCoeffs object
    load_coeffs = SHCoeffs.from_array(coeffs_array, normalization='4pi', csphase=1)

    # Initialize result dictionary
    result = {
        'load_coefficients': load_coeffs,
        'residuals': np.sqrt(residual_norm ** 2 / (3 * n_sites)),
        'rank': len(s_vals[s_vals > threshold]),
        'x_solution': x,
        'singular_values': s_vals,
        'condition_number': cond,
        'design_matrix': A,
        'coefficient_indices': coeff_idx
    }

    # Calculate formal errors if requested
    if calculate_errors:
        if solving_system_method == 'svd':
            try:
                # Compute variance factor
                sigma_0_squared = residual_norm ** 2 / (3 * n_sites - len(x))
                print(f"Estimated error variance factor: {sigma_0_squared:.8f}")

                # Calculate covariance matrix using SVD components
                V_truncated = vh_svd.T[:, s_vals > threshold]
                s_inv_squared = s_inv[s_vals > threshold] ** 2
                cov_matrix = V_truncated @ np.diag(s_inv_squared) @ V_truncated.T * sigma_0_squared

                coefficient_errors = np.sqrt(np.diag(cov_matrix))

                # Initialize error arrays
                error_array = np.zeros_like(coeffs_array)

                # Fill in error arrays
                for (n, m), idx in [(key[:2], val) for key, val in coeff_idx.items() if key[2] == 'C']:
                    error_array[0, n, m] = coefficient_errors[idx]

                for (n, m), idx in [(key[:2], val) for key, val in coeff_idx.items() if key[2] == 'S']:
                    error_array[1, n, m] = coefficient_errors[idx]

                # Signal-to-noise ratio for load coefficients
                snr_load = np.zeros_like(coeffs_array)
                mask = error_array > 0
                snr_load[mask] = np.abs(coeffs_array[mask]) / error_array[mask]

                result.update({
                    'load_errors': error_array,
                    'load_snr': snr_load,
                    'covariance_matrix': cov_matrix,
                    'sigma_0_squared': sigma_0_squared
                })

                print("Computed formal errors for load coefficients")
            except Exception as e:
                print(f"Warning: Could not compute formal errors for load coefficients: {e}")
        elif solving_system_method == 'lsa':
            # Calculate formal errors if requested
            try:
                # Compute variance factor
                sigma_0_squared = residual_norm ** 2 / (3 * n_sites - len(x))
                print(f"Estimated error variance factor: {sigma_0_squared:.8f}")

                # Use eigenvalue decomposition approach by default - more stable for all cases
                print("Computing covariance matrix using stable eigenvalue decomposition...")

                # Compute A^T A
                AT_A = A.T @ A

                # Use eigenvalue decomposition
                eigvals, eigvecs = scipy.linalg.eigh(AT_A)

                # Filter small eigenvalues (regularization)
                max_eigval = eigvals.max()
                min_eigval = max_eigval * 1e-10  # Adjust threshold as needed
                mask = eigvals > min_eigval

                print(f"Using {np.sum(mask)} of {len(eigvals)} eigenvalues (threshold: {min_eigval:.3e})")

                filtered_eigvals = eigvals[mask]
                filtered_eigvecs = eigvecs[:, mask]

                # Construct pseudoinverse using filtered eigendecomposition
                Qxx = filtered_eigvecs @ np.diag(1.0 / filtered_eigvals) @ filtered_eigvecs.T

                # Covariance matrix = sigma_0^2 * Qxx
                cov_matrix = sigma_0_squared * Qxx

                # Extract standard errors from diagonal of covariance matrix
                coefficient_errors = np.sqrt(np.diag(cov_matrix))

                # Check for NaN or inf values
                if np.any(np.isnan(coefficient_errors)) or np.any(np.isinf(coefficient_errors)):
                    print("Warning: NaN or Inf values detected in error estimates.")
                    # Replace NaN/Inf with large but finite values
                    bad_mask = np.isnan(coefficient_errors) | np.isinf(coefficient_errors)
                    coefficient_errors[bad_mask] = 1.0  # Set to a large value relative to coefficients

                # Initialize error arrays
                error_array = np.zeros_like(coeffs_array)

                # Fill in error arrays
                for (n, m), idx in [(key[:2], val) for key, val in coeff_idx.items() if key[2] == 'C']:
                    error_array[0, n, m] = coefficient_errors[idx]

                for (n, m), idx in [(key[:2], val) for key, val in coeff_idx.items() if key[2] == 'S']:
                    error_array[1, n, m] = coefficient_errors[idx]

                # Signal-to-noise ratio for load coefficients
                snr_load = np.zeros_like(coeffs_array)
                mask = error_array > 0
                snr_load[mask] = np.abs(coeffs_array[mask]) / error_array[mask]

                result.update({
                    'load_errors': error_array,
                    'load_snr': snr_load,
                    'covariance_matrix': cov_matrix,
                    'sigma_0_squared': sigma_0_squared
                })

                print("Computed formal errors for load coefficients")
            except Exception as e:
                print(f"Warning: Could not compute formal errors for load coefficients: {e}")
                import traceback
                traceback.print_exc()

    # Print the key coefficients
    print("\nEstimated Key Coefficients (SLD: kg/m**2):")
    print(f"C10 = {coeffs_array[0, 1, 0]:.8e}")
    print(f"C20 = {coeffs_array[0, 2, 0]:.8e}")
    print(f"C21 = {coeffs_array[0, 2, 1]:.8e}, S21 = {coeffs_array[1, 2, 1]:.8e}")
    print(f"C22 = {coeffs_array[0, 2, 2]:.8e}, S22 = {coeffs_array[1, 2, 2]:.8e}")
    print(f"C30 = {coeffs_array[0, 3, 0]:.8e}")

    return result
def calculate_potential_coefficients(load_coeffs, love_numbers_file=None, error_info=None, reference_frame="CF"):
    """
    Compute gravity potential coefficients from load coefficients.

    Parameters:
    -----------
    load_coeffs : SHCoeffs
        Load coefficients object
    love_numbers_file : str, optional
        Path to file containing Love numbers
    error_info : dict, optional
        Dictionary containing error information for load coefficients
    reference_frame : str, optional
        Reference frame to use for calculations. Options:
            "CE" - Center of mass of the solid Earth
            "CM" - Center of mass of the Earth system
            "CF" - Center of surface figure (no-net translation)
            "CL" - Center of surface lateral figure (no-net horizontal translation)
            "CH" - Center of surface height figure (no-net vertical translation)
        Default: "CE"

    Returns:
    --------
    dict
        Dictionary containing potential coefficients and error information if provided
    """
    # Constants
    R_E = 6371e3  # Earth radius (m)
    rho_w = 1025.0  # Density of seawater (kg/m³)
    rho_E = 5517.0  # Average density of solid Earth (kg/m^3)
    M_E = 5.9722 * 10**24
    # Load Love numbers (in CE frame)
    love_numbers = load_love_numbers(love_numbers_file)

    # Transform degree-1 load Love numbers based on reference frame
    transformed_love_numbers = transform_love_numbers(love_numbers, reference_frame)

    # Get maximum degree from load coefficients
    max_degree = load_coeffs.coeffs.shape[1] - 1

    # Initialize result
    result = {}

    # Convert load coefficients to potential coefficients
    potential_array = np.zeros_like(load_coeffs.coeffs)

    # Calculate potential coefficients
    for n in range(1, max_degree + 1):
        k_n = transformed_love_numbers['k_n'][n]  # Gravitational load Love number
        # Factor from Wahr et al. (1998) and papers, scaled by rho_w
        # The factor 3.0/(rho_E * R_E) includes the normalization and unit conversion
        factor = 3.0 / (rho_E * R_E) * (1.0 + k_n) / (2.0 * n + 1.0)
        # factor = (1.0 + k_n) / (2.0 * n + 1.0)
        # factor = (1+k_n) * (3*rho_w)/(rho_E*(2*n+1)) * (R_E/M_E)
        potential_array[:, n, :] = factor * load_coeffs.coeffs[:, n, :]

    # Create SHCoeffs object for potential coefficients
    potential_coeffs = SHCoeffs.from_array(potential_array, normalization='4pi', csphase=1)
    result['potential_coefficients'] = potential_coeffs

    # Calculate errors if error information is provided
    if error_info is not None and 'load_errors' in error_info:
        try:
            load_errors = error_info['load_errors']
            potential_error_array = np.zeros_like(load_errors)

            # Calculate potential coefficient errors
            for n in range(1, max_degree + 1):
                k_n = transformed_love_numbers['k_n'][n]  # Gravitational load Love number
                factor = 3.0 / (rho_E * R_E) * (1.0 + k_n) / (2.0 * n + 1.0)
                potential_error_array[:, n, :] = factor * load_errors[:, n, :]

            # Signal-to-noise ratio for potential coefficients
            snr_potential = np.zeros_like(potential_array)

            # Avoid division by zero
            mask_potential = potential_error_array > 0
            snr_potential[mask_potential] = np.abs(potential_array[mask_potential]) / potential_error_array[
                mask_potential]

            # Add to results
            result.update({
                'potential_errors': potential_error_array,
                'potential_snr': snr_potential
            })

            print("Computed formal errors for potential coefficients")
        except Exception as e:
            print(f"Warning: Could not compute formal errors for potential coefficients: {e}")

    return result

def compute_gravity_field_coefficients(displacements, max_degree=60, love_numbers_file=None, calculate_errors=False,
                                      reference_frame="CE", verify_solution=True, save_summary=False,
                                      output_dir=None, prefix="gravity_coeffs", identifier=None,
                                      regularization=False, add_helmert=False):
    """
    Convert site displacements into gravity field coefficients using unified least squares adjustment.

    Parameters:
    -----------
    displacements : dict
        Dictionary containing:
            'vertical': list of vertical displacements (m)
            'east': list of eastward displacements (m)
            'north': list of northward displacements (m)
            'lat': list of latitudes (degrees)
            'lon': list of longitudes (degrees)
    max_degree : int
        Maximum spherical harmonic degree
    love_numbers_file : str, optional
        Path to file containing Love numbers
    calculate_errors : bool, optional
        Whether to calculate formal errors for the coefficients (default: False)
    reference_frame : str, optional
        Reference frame to use for calculations. Options:
            "CE" - Center of mass of the solid Earth
            "CM" - Center of mass of the Earth system
            "CF" - Center of surface figure (no-net translation)
            "CL" - Center of surface lateral figure (no-net horizontal translation)
            "CH" - Center of surface height figure (no-net vertical translation)
        Default: "CE"
    save_summary : bool, optional
        Whether to save processing summary in machine-readable formats
    output_dir : str, optional
        Directory to save summary files (required if save_summary=True)
    prefix : str, optional
        Prefix for output files
    identifier : str, optional
        Additional identifier to include in filenames

    Returns:
    --------
    coeffs : dict
        Dictionary containing load and gravity potential coefficients
    """
    # Step 1: Calculate load coefficients
    load_result = calculate_load_coefficients(
        displacements,
        max_degree=max_degree,
        love_numbers_file=love_numbers_file,
        calculate_errors=calculate_errors,
        reference_frame=reference_frame,
        regularize=regularization,
        add_helmert=add_helmert
    )

    # power_spectrum = load_result['load_coefficients'].spectrum(unit='per_l')

    # Step 2: Calculate potential coefficients
    error_info = None
    if calculate_errors and 'load_errors' in load_result:
        error_info = {
            'load_errors': load_result['load_errors'],
            'sigma_0_squared': load_result.get('sigma_0_squared', 1.0)
        }

    potential_result = calculate_potential_coefficients(
        load_result['load_coefficients'],
        love_numbers_file=love_numbers_file,
        error_info=error_info,
        reference_frame=reference_frame
    )

    # Step 3: Combine results
    final_result = {
        'load_coefficients': load_result['load_coefficients'],
        'potential_coefficients': potential_result['potential_coefficients'],
        'residuals': load_result['residuals'],
        'rank': load_result['rank']
    }

    # Add error information if available
    if calculate_errors:
        if 'load_errors' in load_result:
            final_result['load_errors'] = load_result['load_errors']
            final_result['load_snr'] = load_result['load_snr']

        if 'potential_errors' in potential_result:
            final_result['potential_errors'] = potential_result['potential_errors']
            final_result['potential_snr'] = potential_result['potential_snr']

        if 'sigma_0_squared' in load_result:
            final_result['sigma_0_squared'] = load_result['sigma_0_squared']

        if 'covariance_matrix' in load_result:
            final_result['covariance_matrix'] = load_result['covariance_matrix']

    if verify_solution:
        print("\nVerifying solution with forward modeling...")
        validation = verify_gravity_solution(
            displacements,
            final_result,
            love_numbers_file,
            reference_frame
        )

        # Add validation results to the output
        final_result['validation'] = validation

        # If validation metrics are poor, print warning
        if validation['variance_explained']['up'] < 0.75:
            print("WARNING: Poor fit in vertical component! Consider:")
            print("  - Checking data quality")
            print("  - Reducing maximum degree")
            print("  - Adding regularization")

    if save_summary:
        if output_dir is None:
            print("Warning: output_dir not specified, summary will not be saved")
        else:
            # Add reference frame to the result dictionary for inclusion in summary
            final_result['reference_frame'] = reference_frame
            final_result['max_degree'] = max_degree

            # Save the summary
            summary_files = save_processing_summary(
                final_result,
                output_dir=output_dir,
                prefix=prefix,
                identifier=identifier,
                formats=['yaml']
            )

            # Add summary file paths to the result
            final_result['summary_files'] = summary_files

    return final_result


def transform_love_numbers(love_numbers, reference_frame="CE"):
    """
    Transform Love numbers from CE frame to the target reference frame.

    Following Blewitt (2003): "Self-consistency in reference frames, geocenter definition,
    and surface loading of the solid Earth"

    Parameters:
    -----------
    love_numbers : dict
        Dictionary containing Love numbers in CE frame
    reference_frame : str
        Target reference frame: "CE", "CM", "CF", "CL", or "CH"

    Returns:
    --------
    dict
        Dictionary containing transformed Love numbers
    """
    # Make a copy of the love numbers to avoid modifying the original
    transformed = {key: value.copy() if isinstance(value, (list, np.ndarray)) else value
                   for key, value in love_numbers.items()}

    # If the reference frame is already CE, no transformation needed
    if reference_frame == "CE":
        return transformed

    # Extract degree-1 Love numbers in CE frame
    h1_CE = transformed['h_n'][1]
    l1_CE = transformed['l_n'][1]
    k1_CE = transformed['k_n'][1]

    # Transform degree-1 Love numbers based on reference frame
    if reference_frame == "CM":
        # Center of mass of the entire Earth system
        transformed['h_n'][1] = h1_CE - 1.0
        transformed['l_n'][1] = l1_CE - 1.0
        transformed['k_n'][1] = -1.0  # Makes (1+k'₁)=0 in CM frame

    elif reference_frame == "CF":
        # Center of surface figure (no-net translation)
        transformed['h_n'][1] = (2.0 / 3.0) * (h1_CE - l1_CE)
        transformed['l_n'][1] = -(1.0 / 3.0) * (h1_CE - l1_CE)
        transformed['k_n'][1] = k1_CE - (1.0 / 3.0) * h1_CE - (2.0 / 3.0) * l1_CE

    else:
        raise ValueError(f"Unknown reference frame: {reference_frame}. "
                         f"Valid options are: CE, CM, CF")

    return transformed

def load_love_numbers(filename=None, silent=False):
    """
    Load Love numbers from a file or use approximate values.

    Parameters:
    -----------
    filename : str, optional
        Path to file containing Love numbers

    Returns:
    --------
    love_numbers : dict
        Dictionary containing Love numbers k'n, h'n, l'n
    """
    if filename is None:
        # Use approximate values based on PREM model
        max_degree = 60
        degrees = np.arange(max_degree + 1)

        # Approximate formulations, should be replaced with actual values
        k_n = 0.3 * np.exp(-0.05 * degrees)  # k'n (potential)
        h_n = 0.6 * np.exp(-0.05 * degrees)  # h'n (vertical)
        l_n = 0.08 * np.exp(-0.02 * degrees)  # l'n (horizontal)

        if not silent:
            print("Using approximate Love numbers based on PREM model.")

        return {
            'k_n': k_n,
            'h_n': h_n,
            'l_n': l_n
        }
    else:
        # Load from file - implementation depends on file format
        try:
            data = np.loadtxt(filename, skiprows=1, max_rows=300)  # Assuming a simple format with header
            degrees = data[:, 0].astype(int)
            max_degree = degrees[-1]

            k_n = np.zeros(max_degree + 1)
            h_n = np.zeros(max_degree + 1)
            l_n = np.zeros(max_degree + 1)

            h_n[degrees] = data[:, 1]  # Assuming column 1 contains h'n
            l_n[degrees] = data[:, 2]  # Assuming column 2 contains l'n
            k_n[degrees] = data[:, 3]  # Assuming column 3 contains k'n

            love_numbers = {
                'k_n': k_n,
                'h_n': h_n,
                'l_n': l_n
            }
        except Exception as e:
            print(f"Error loading Love numbers: {e}")
            print("Using approximate values instead")
            return load_love_numbers(None)

        return love_numbers

def prepare_displacements_from_df(df):
    """
    Prepare displacement dictionary from dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing displacement data with columns:
        'DN', 'DE', 'DU', 'Latitude', 'Longitude'

    Returns:
    --------
    displacements : dict
        Dictionary containing displacement and site location data
    """
    # Extract displacement data
    north_disp = df['dN'].values
    east_disp = df['dE'].values
    up_disp = df['dU'].values

    # Extract station coordinates
    lats = df['Latitude'].values
    lons = df['Longitude'].values

    # Check for NaN values
    import numpy as np
    mask = ~(np.isnan(north_disp) | np.isnan(east_disp) | np.isnan(up_disp) |
             np.isnan(lats) | np.isnan(lons))

    if not np.all(mask):
        print(f"Warning: Found {np.sum(~mask)} sites with NaN values. These will be excluded.")

        north_disp = north_disp[mask]
        east_disp = east_disp[mask]
        up_disp = up_disp[mask]
        lats = lats[mask]
        lons = lons[mask]

    # Organize displacements into dictionary
    displacements = {
        'vertical': up_disp*1e-3,
        'east': east_disp*1e-3,
        'north': north_disp*1e-3,
        'lat': lats,
        'lon': lons
    }

    return displacements

def export_coefficients(coeffs, output_dir, prefix="gravity_coeffs", identifier=None, icgem_format=False):
    """
    Export spherical harmonic coefficients to files, with option to use PyShTools for ICGEM format.

    Parameters:
    -----------
    coeffs : dict
        Dictionary containing load and gravity potential coefficients
    output_dir : str
        Directory to save coefficient files
    prefix : str, optional
        Prefix for output files
    identifier : str, optional
        Additional identifier to include in filenames
    icgem_format : bool, optional
        Whether to also export in ICGEM format using PyShTools

    Returns:
    --------
    file_dict : dict
        Dictionary with paths to the exported files
    """
    import os
    import numpy as np
    import datetime
    import pyshtools as pysh

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create filename prefix
    file_prefix = f"{prefix}" if not identifier else f"{prefix}_{identifier}"

    # Check if coeffs is a dictionary with the expected keys
    if not isinstance(coeffs, dict) or 'load_coefficients' not in coeffs or 'potential_coefficients' not in coeffs:
        raise TypeError("Expected a dictionary with 'load_coefficients' and 'potential_coefficients' keys. "
                        f"Got type: {type(coeffs)}")

    # Export load coefficients
    load_file = os.path.join(output_dir, f"{file_prefix}_load.txt")

    if not hasattr(coeffs['load_coefficients'], 'coeffs'):
        raise TypeError("Expected an SHCoeffs object for 'load_coefficients' but got a different type.")

    c_array = coeffs['load_coefficients'].coeffs
    max_degree = c_array.shape[1] - 1

    # Check if error information is available
    has_errors = 'load_errors' in coeffs and coeffs['load_errors'] is not None
    has_snr = 'load_snr' in coeffs and coeffs['load_snr'] is not None

    with open(load_file, 'w') as f:
        f.write(f"# Load spherical harmonic coefficients\n")

        if has_errors and has_snr:
            f.write("# n m C_nm S_nm sigma_C_nm sigma_S_nm SNR_C SNR_S\n")
        else:
            f.write("# n m C_nm S_nm\n")

        for n in range(max_degree + 1):
            for m in range(n + 1):
                C_nm = c_array[0, n, m]
                S_nm = c_array[1, n, m] if m > 0 else 0.0

                if has_errors and has_snr:
                    sigma_C = coeffs['load_errors'][0, n, m]
                    sigma_S = coeffs['load_errors'][1, n, m] if m > 0 else 0.0
                    snr_C = coeffs['load_snr'][0, n, m]
                    snr_S = coeffs['load_snr'][1, n, m] if m > 0 else 0.0
                    f.write(f"{n} {m} {C_nm:.16e} {S_nm:.16e} {sigma_C:.16e} {sigma_S:.16e} {snr_C:.4f} {snr_S:.4f}\n")
                else:
                    f.write(f"{n} {m} {C_nm:.16e} {S_nm:.16e}\n")

    print(f"Exported load coefficients to {load_file}")

    # Export potential coefficients
    pot_file = os.path.join(output_dir, f"{file_prefix}_potential.txt")

    if not hasattr(coeffs['potential_coefficients'], 'coeffs'):
        raise TypeError("Expected an SHCoeffs object for 'potential_coefficients' but got a different type.")

    p_array = coeffs['potential_coefficients'].coeffs

    # Check if error information is available for potential coefficients
    has_pot_errors = 'potential_errors' in coeffs and coeffs['potential_errors'] is not None
    has_pot_snr = 'potential_snr' in coeffs and coeffs['potential_snr'] is not None

    with open(pot_file, 'w') as f:
        f.write(f"# Gravity potential spherical harmonic coefficients\n")

        if has_pot_errors and has_pot_snr:
            f.write("# n m C_nm S_nm sigma_C_nm sigma_S_nm SNR_C SNR_S\n")
        else:
            f.write("# n m C_nm S_nm\n")

        for n in range(max_degree + 1):
            for m in range(n + 1):
                C_nm = p_array[0, n, m]
                S_nm = p_array[1, n, m] if m > 0 else 0.0

                if has_pot_errors and has_pot_snr:
                    sigma_C = coeffs['potential_errors'][0, n, m]
                    sigma_S = coeffs['potential_errors'][1, n, m] if m > 0 else 0.0
                    snr_C = coeffs['potential_snr'][0, n, m]
                    snr_S = coeffs['potential_snr'][1, n, m] if m > 0 else 0.0
                    f.write(f"{n} {m} {C_nm:.16e} {S_nm:.16e} {sigma_C:.16e} {sigma_S:.16e} {snr_C:.4f} {snr_S:.4f}\n")
                else:
                    f.write(f"{n} {m} {C_nm:.16e} {S_nm:.16e}\n")

    print(f"Exported potential coefficients to {pot_file}")

    # Export in ICGEM format if requested
    load_icgem_file = None
    pot_icgem_file = None

    if icgem_format:
        top_header = f'''
        GNSS-based gravity model
        Created: {datetime.datetime.now().strftime('%Y-%m-%d')}
        '''

        header = {
            'modelname': f"{file_prefix}",
            'product_type': 'gravity_field',
            'earth_gm': 3.986004418e+14,  # GM in m³/s²
            'r0': 6.3710e+06,  # R in m
            'lmax': max_degree,
            'tide_system': 'zero_tide',  # Assuming zero tide system
            'error_kind': 'formal',
            'normalization': '4pi'
        }
        # Create load SHGravCoeffs object for PyShTools
        load_icgem_file = os.path.join(output_dir, f"{file_prefix}_load.gfc")

        # Convert SHCoeffs to SHGravCoeffs
        try:
            header['modelname'] = f"{file_prefix}_load"

            # Export to ICGEM format
            pysh.shio.write_icgem_gfc(load_icgem_file,
                                 coeffs['load_coefficients'].coeffs,
                                 coeffs['load_errors'],
                                 **header)

            print(f"Exported load coefficients in ICGEM format to {load_icgem_file}")

        except Exception as e:
            print(f"Error using PyShTools to export load coefficients: {e}")
            # Fallback to manual ICGEM export

        # ICGEM format for potential coefficients using PyShTools
        pot_icgem_file = os.path.join(output_dir, f"{file_prefix}_potential.gfc")

        try:
            header['modelname'] = f"{file_prefix}_potential"
            # Export to ICGEM format
            pysh.shio.write_icgem_gfc(pot_icgem_file,
                                      coeffs['potential_coefficients'].coeffs,
                                      coeffs['potential_errors'],
                                      **header)

            print(f"Exported potential coefficients in ICGEM format to {pot_icgem_file}")

        except Exception as e:
            print(f"Error using PyShTools to export potential coefficients: {e}")

    # Export summary if error information is available
    summary_file = None
    if 'sigma_0_squared' in coeffs or 'residuals' in coeffs:
        summary_file = os.path.join(output_dir, f"{file_prefix}_summary.txt")

        with open(summary_file, 'w') as f:
            f.write(f"# Spherical harmonic inversion summary\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Maximum degree: {max_degree}\n")

            if 'rank' in coeffs:
                f.write(f"Rank of design matrix: {coeffs['rank']}\n")

            if 'sigma_0_squared' in coeffs:
                f.write(f"Variance factor (sigma_0^2): {coeffs['sigma_0_squared']:.8e}\n")

            if 'residuals' in coeffs:
                f.write(f"RMS of residuals: {coeffs['residuals']:.8e} m\n\n")

            # Add degree variance analysis if errors are available
            if has_errors:
                f.write("# Degree variance analysis\n")
                f.write("# n load_power load_error_power SNR\n")

                for n in range(1, max_degree + 1):
                    # Calculate degree variance (power per degree)
                    load_power = 0
                    load_error = 0

                    for m in range(n + 1):
                        # Load coefficients power
                        load_power += c_array[0, n, m] ** 2
                        if m > 0:
                            load_power += c_array[1, n, m] ** 2

                        # Load coefficient errors power
                        if has_errors:
                            load_error += coeffs['load_errors'][0, n, m] ** 2
                            if m > 0:
                                load_error += coeffs['load_errors'][1, n, m] ** 2

                    snr = np.sqrt(load_power / load_error) if load_error > 0 else float('inf')
                    f.write(f"{n} {load_power:.8e} {load_error:.8e} {snr:.4f}\n")

        print(f"Exported inversion summary to {summary_file}")

    return {
        'load_file': load_file,
        'potential_file': pot_file,
        'load_icgem_file': load_icgem_file,
        'pot_icgem_file': pot_icgem_file,
        'summary_file': summary_file
    }


def verify_gravity_solution(displacements, coeffs, love_numbers_file=None, reference_frame="CE"):
    """
    Verify the gravity field solution by comparing forward-modeled displacements
    with original displacements.

    Parameters:
    -----------
    displacements : dict
        Original displacements
    coeffs : dict
        Coefficient solution
    love_numbers_file : str, optional
        Path to Love numbers file
    reference_frame : str, optional
        Reference frame

    Returns:
    --------
    dict
        Dictionary with validation metrics
    """

    # Perform the validation
    validation = validate_coefficient_solution(
        displacements,
        coeffs,
        love_numbers_file,
        reference_frame
    )

    # Analyze the spectrum if needed
    # from gravity_validation import analyze_coefficient_spectrum
    # spectrum = analyze_coefficient_spectrum(coeffs)

    return validation


def save_processing_summary(coeffs, output_dir, prefix="gravity_coeffs", identifier=None, formats=None):
    """
    Save a processing summary in machine-readable formats.

    Parameters:
    -----------
    coeffs : dict
        Dictionary containing the coefficient solution and validation information
    output_dir : str
        Directory to save the summary files
    prefix : str, optional
        Prefix for output files
    identifier : str, optional
        Additional identifier to include in filenames
    formats : list, optional
        List of output formats, defaults to ['json', 'yaml']

    Returns:
    --------
    dict
        Dictionary with paths to the exported files
    """
    import os
    import json
    import datetime
    import numpy as np
    from pathlib import Path

    # Default formats
    if formats is None:
        formats = ['json', 'yaml']

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create filename prefix
    file_prefix = f"{prefix}" if not identifier else f"{prefix}_{identifier}"

    # Extract and format summary information
    summary = {
        "processing_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "reference_frame": coeffs.get("reference_frame", "unknown"),
        "max_degree": coeffs.get("max_degree", -1),
    }

    # Add solution rank and fit statistics
    if 'rank' in coeffs:
        summary["rank"] = int(coeffs['rank'])

    if 'residuals' in coeffs:
        summary["rms_residuals"] = float(coeffs['residuals'])

    if 'sigma_0_squared' in coeffs:
        summary["sigma_0_squared"] = float(coeffs['sigma_0_squared'])

    # Add validation metrics if available
    if 'validation' in coeffs:
        validation = {}

        # RMS errors
        if 'rms' in coeffs['validation']:
            validation["rms"] = {
                "north": float(coeffs['validation']['rms']['north']),
                "east": float(coeffs['validation']['rms']['east']),
                "up": float(coeffs['validation']['rms']['up']),
                "total": float(coeffs['validation']['rms']['total'])
            }

        # Correlation
        if 'correlation' in coeffs['validation']:
            validation["correlation"] = {
                "north": float(coeffs['validation']['correlation']['north']),
                "east": float(coeffs['validation']['correlation']['east']),
                "up": float(coeffs['validation']['correlation']['up'])
            }

        # Variance explained
        if 'variance_explained' in coeffs['validation']:
            validation["variance_explained"] = {
                "north": float(coeffs['validation']['variance_explained']['north']),
                "east": float(coeffs['validation']['variance_explained']['east']),
                "up": float(coeffs['validation']['variance_explained']['up'])
            }

        # Signal-to-residual ratio
        if 'signal_to_residual' in coeffs['validation']:
            validation["signal_to_residual"] = {
                "north": float(coeffs['validation']['signal_to_residual']['north']),
                "east": float(coeffs['validation']['signal_to_residual']['east']),
                "up": float(coeffs['validation']['signal_to_residual']['up'])
            }

        summary["validation"] = validation

    # Add degree variance information
    if 'load_coefficients' in coeffs and hasattr(coeffs['load_coefficients'], 'coeffs'):
        c_array = coeffs['load_coefficients'].coeffs
        max_degree = c_array.shape[1] - 1

        has_errors = 'load_errors' in coeffs and coeffs['load_errors'] is not None

        degree_variance = []
        for n in range(1, max_degree + 1):
            # Calculate degree variance (power per degree)
            load_power = 0
            load_error = 0

            for m in range(n + 1):
                # Load coefficients power
                load_power += c_array[0, n, m] ** 2
                if m > 0:
                    load_power += c_array[1, n, m] ** 2

                # Load coefficient errors power
                if has_errors:
                    load_error += coeffs['load_errors'][0, n, m] ** 2
                    if m > 0:
                        load_error += coeffs['load_errors'][1, n, m] ** 2

            snr = np.sqrt(load_power / load_error) if (has_errors and load_error > 0) else float('inf')
            degree_variance.append({
                "degree": n,
                "power": float(load_power),
                "error_power": float(load_error) if has_errors else None,
                "snr": float(snr) if not np.isinf(snr) else "inf"
            })

        summary["degree_variance"] = degree_variance

    # Add coefficient statistics by degree
    if 'load_coefficients' in coeffs and hasattr(coeffs['load_coefficients'], 'coeffs'):
        c_array = coeffs['load_coefficients'].coeffs
        max_degree = c_array.shape[1] - 1

        has_errors = 'load_errors' in coeffs and coeffs['load_errors'] is not None

        coeff_stats = {}
        for n in range(1, max_degree + 1):
            degree_coeffs = []
            for m in range(n + 1):
                c_nm = float(c_array[0, n, m])
                s_nm = float(c_array[1, n, m]) if m > 0 else 0.0

                coeff_info = {
                    "C": c_nm,
                    "S": s_nm if m > 0 else None
                }

                if has_errors:
                    sigma_c = float(coeffs['load_errors'][0, n, m])
                    sigma_s = float(coeffs['load_errors'][1, n, m]) if m > 0 else 0.0

                    coeff_info["sigma_C"] = sigma_c
                    coeff_info["sigma_S"] = sigma_s if m > 0 else None

                degree_coeffs.append(coeff_info)

            coeff_stats[f"degree_{n}"] = degree_coeffs

        summary["coefficient_statistics"] = coeff_stats

    # Export to different formats
    output_files = {}

    # JSON export
    if 'json' in formats:
        json_file = os.path.join(output_dir, f"{file_prefix}_summary.json")
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        output_files['json'] = json_file
        print(f"Saved JSON summary to {json_file}")

    # YAML export
    if 'yaml' in formats:
        try:
            import yaml
            yaml_file = os.path.join(output_dir, f"{file_prefix}_summary.yaml")
            with open(yaml_file, 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)
            output_files['yaml'] = yaml_file
            print(f"Saved YAML summary to {yaml_file}")
        except ImportError:
            print("PyYAML not installed. Skipping YAML export.")

    # CSV export (for selected summary metrics)
    if 'csv' in formats:
        import csv
        csv_file = os.path.join(output_dir, f"{file_prefix}_summary.csv")

        # Extract key metrics for CSV
        flat_metrics = {
            "processing_time": summary["processing_time"],
            "reference_frame": summary.get("reference_frame", "unknown"),
            "max_degree": summary.get("max_degree", -1),
            "rank": summary.get("rank", "N/A"),
            "rms_residuals": summary.get("rms_residuals", "N/A")
        }

        # Add validation metrics if available
        if "validation" in summary:
            validation = summary["validation"]
            if "rms" in validation:
                for key, value in validation["rms"].items():
                    flat_metrics[f"rms_{key}"] = value

            if "variance_explained" in validation:
                for key, value in validation["variance_explained"].items():
                    flat_metrics[f"var_exp_{key}"] = value

        # Write to CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(flat_metrics.keys())
            # Write values
            writer.writerow(flat_metrics.values())

        output_files['csv'] = csv_file
        print(f"Saved CSV summary to {csv_file}")

    return output_files


def add_tikhonov_regularization(A, y, coeff_indices, max_degree, damping_factor=0.01,
                               degree_dependent=True, order_dependent=False):
    """
    Add Tikhonov regularization to stabilize the inversion of gravity field coefficients.
    """
    import numpy as np

    # Get dimensions of the original system
    n_rows, n_cols = A.shape

    # Create regularization matrix (initially empty)
    R = np.zeros((0, n_cols))
    d = np.zeros(0)  # Right-hand side for regularization (zeros for standard Tikhonov)

    # Group coefficients by degree for convenience
    degree_coeffs = {}
    for (n, m, cs), idx in coeff_indices.items():
        if n not in degree_coeffs:
            degree_coeffs[n] = []
        degree_coeffs[n].append((m, cs, idx))

    # Apply regularization for each degree
    for n in range(1, max_degree + 1):
        if n not in degree_coeffs:
            continue

        # Get all coefficients for this degree
        coeffs = degree_coeffs[n]
        n_coeffs = len(coeffs)

        # Compute degree-dependent scaling if requested
        if degree_dependent:
            degree_scale = n ** 2  # Stronger regularization for higher degrees
        else:
            degree_scale = 1.0

        # Create regularization block for this degree
        R_n = np.zeros((n_coeffs, n_cols))

        # Apply regularization to each coefficient
        for i, (m, cs, idx) in enumerate(coeffs):
            # Compute order-dependent scaling if requested
            if order_dependent:
                order_scale = 1.0 + m ** 2  # More regularization for tesseral/sectoral terms
            else:
                order_scale = 1.0

            # Set regularization weight for this coefficient
            weight = damping_factor * degree_scale * order_scale
            R_n[i, idx] = weight

        # Add to regularization matrix
        R = np.vstack((R, R_n))
        d = np.append(d, np.zeros(n_coeffs))

    # Augment the design matrix and observation vector
    A_aug = np.vstack((A, R))
    y_aug = np.append(y, d)  # Include the original observations!

    # Return the augmented system
    return A_aug, y_aug


def find_optimal_damping_factor(A, y, coeff_indices, max_degree,
                                range_min=1e-10, range_max=1e-3, n_values=20,
                                degree_dependent=True):
    """
    Find optimal damping factor using Generalized Cross-Validation (GCV).
    """
    import numpy as np

    # Generate logarithmically spaced damping factors to test
    damping_factors = np.logspace(np.log10(range_min), np.log10(range_max), n_values)

    # Store GCV scores
    gcv_scores = np.zeros(n_values)

    # Test each damping factor
    for i, damping in enumerate(damping_factors):
        # Apply regularization
        A_reg, y_reg = add_tikhonov_regularization(
            A, y, coeff_indices, max_degree,
            damping_factor=damping,
            degree_dependent=degree_dependent
        )

        # Solve the regularized system
        try:
            x_reg = np.linalg.lstsq(A_reg, y_reg, rcond=None)[0]

            # Compute residuals for original system
            residuals = y - A @ x_reg

            # Calculate effective degrees of freedom
            # Use a simpler approximation for efficiency
            n = len(y)
            p_eff = np.linalg.matrix_rank(A)

            # Compute GCV score
            gcv_scores[i] = n * np.sum(residuals ** 2) / (n - p_eff) ** 2

            print(f"Testing damping factor {damping:.10f}: GCV score = {gcv_scores[i]:.6e}")
        except np.linalg.LinAlgError:
            print(f"Numerical error for damping factor {damping:.10f}, skipping")
            gcv_scores[i] = np.inf

    # Find optimal damping factor (among finite scores)
    valid_indices = np.isfinite(gcv_scores)
    if np.any(valid_indices):
        best_idx = np.argmin(gcv_scores[valid_indices])
        optimal_damping = damping_factors[valid_indices][best_idx]
        print(
            f"Optimal damping factor = {optimal_damping:.10e} (GCV score = {gcv_scores[valid_indices][best_idx]:.6e})")
    else:
        # Default to the smallest damping factor if all GCV calculations failed
        optimal_damping = range_min
        print(f"All GCV calculations failed, defaulting to damping factor = {optimal_damping:.10e}")

    return optimal_damping