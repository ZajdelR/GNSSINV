# toolbox_gravity_validation.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import scipy.special
from pyshtools import SHCoeffs
import datetime


def calculate_displacements_from_coefficients(coeffs, site_locations,
                                              love_numbers_file=None,
                                              reference_frame="CF"):
    """
    Calculate site displacements from load coefficients via forward modeling.
    Compatible with coefficients from `calculate_load_coefficients`.

    Parameters
    ----------
    coeffs : SHCoeffs object or dict
        Spherical harmonic coefficients representing surface mass load
    site_locations : dict
        Dictionary containing site coordinates with 'lat' and 'lon' keys
    love_numbers_file : str, optional
        Path to a file containing load Love numbers
    reference_frame : str, optional
        Reference frame for Love numbers, default is "CF"

    Returns
    -------
    dict
        Dictionary containing displacement components (north, east, vertical)
    """
    import numpy as np
    import scipy.special
    from toolbox_displacement_to_gravity_coefficients import load_love_numbers, transform_love_numbers

    # Constants
    R_E = 6371000.0  # Earth radius in meters
    rho_E = 5517.0  # Average Earth density (kg/m³)
    M_E = 5.972e24

    # Extract coefficient array
    if isinstance(coeffs, dict) and 'load_coefficients' in coeffs:
        coeff_array = coeffs['load_coefficients'].coeffs
    else:
        coeff_array = coeffs.coeffs

    # Site locations
    if 'lat' in site_locations and 'lon' in site_locations:
        lats = np.array(site_locations['lat'])
        lons = np.array(site_locations['lon'])
    elif 'Latitude' in site_locations and 'Longitude' in site_locations:
        lats = np.array(site_locations['Latitude'])
        lons = np.array(site_locations['Longitude'])
    else:
        raise KeyError("Site locations must contain either 'lat'/'lon' or 'Latitude'/'Longitude'.")

    phi = np.radians(lats)
    lam = np.radians(lons)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Love numbers
    love_numbers = load_love_numbers(love_numbers_file)
    ln = transform_love_numbers(love_numbers, reference_frame)
    h_n = ln['h_n']
    l_n = ln['l_n']
    k_n = ln['k_n']

    print(f"Using reference frame: {reference_frame}")
    print(f"Degree-1 load Love numbers: h'1={h_n[1]:.7f}, l'1={l_n[1]:.7f}, k'1={k_n[1]:.7f}")

    # Prepare result arrays
    n_sites = len(phi)
    north = np.zeros(n_sites)
    east = np.zeros(n_sites)
    up = np.zeros(n_sites)

    max_degree = coeff_array.shape[1] - 1

    for i in range(n_sites):
        phi_i = phi[i]
        lam_i = lam[i]
        cos_phi_i = cos_phi[i]
        sin_phi_i = sin_phi[i]
        z = np.clip(np.cos(phi_i), -1 + 1e-10, 1 - 1e-10)  # Clipped for stability

        for n in range(1, max_degree + 1):
            # Use lpmn but with log-based normalization for stability
            Pnm, dPnm = scipy.special.lpmn(n, n, z)

            for m in range(0, n + 1):
                P = Pnm[m, n]
                dP = dPnm[m, n]

                # Full 4π normalization with log-based calculation
                if 1:
                    if m == 0:
                        norm_factor = np.sqrt(2 * n + 1)
                    else:
                        # More stable calculation using log factorials
                        log_norm = 0.5 * (np.log(2) + np.log(2 * n + 1) +
                                          scipy.special.gammaln(n - m + 1) -
                                          scipy.special.gammaln(n + m + 1))
                        norm_factor = np.exp(log_norm)

                    P_norm = P * norm_factor
                    dP_dz_norm = dP * norm_factor

                dP_dphi = -sin_phi_i * dP_dz_norm

                cos_m_lam = np.cos(m * lam_i)
                sin_m_lam = np.sin(m * lam_i)

                C = coeff_array[0, n, m]
                S = coeff_array[1, n, m] if m > 0 else 0.0

                factor = (rho_E / 3.0) * ((2 * n + 1) / (1.0 + k_n[n]))

                # NORTH
                north_factor = -factor * l_n[n] * dP_dphi / R_E
                north[i] += north_factor * C * cos_m_lam
                if m > 0:
                    north[i] += north_factor * S * sin_m_lam

                # EAST
                if abs(cos_phi_i) > 1e-10:
                    east_factor = -factor * l_n[n] * (m / cos_phi_i) * P_norm / R_E
                    east[i] += -east_factor * C * sin_m_lam
                    if m > 0:
                        east[i] += east_factor * S * cos_m_lam

                # VERTICAL
                up_factor = factor * h_n[n] * P_norm / R_E
                up[i] += up_factor * C * cos_m_lam
                if m > 0:
                    up[i] += up_factor * S * sin_m_lam

    return {
        'north': north,
        'east': east,
        'vertical': up,
        'lat': lats,
        'lon': lons
    }



def validate_coefficient_solution(original_displacements, coeffs, love_numbers_file=None, reference_frame="CE"):
    """
    Validate coefficient solution by comparing original and reconstructed displacements.
    """
    # Calculate reconstructed displacements
    site_locations = {
        'lat': original_displacements['lat'],
        'lon': original_displacements['lon']
    }

    reconstructed = calculate_displacements_from_coefficients(
        coeffs,
        site_locations,
        love_numbers_file,
        reference_frame
    )

    # Calculate residuals
    n_residuals = original_displacements['north'] - reconstructed['north']
    e_residuals = original_displacements['east'] - reconstructed['east']
    u_residuals = original_displacements['vertical'] - reconstructed['vertical']

    # Calculate RMS errors
    n_rms = np.sqrt(np.mean(n_residuals ** 2))
    e_rms = np.sqrt(np.mean(e_residuals ** 2))
    u_rms = np.sqrt(np.mean(u_residuals ** 2))
    total_rms = np.sqrt(np.mean(n_residuals ** 2 + e_residuals ** 2 + u_residuals ** 2))

    # Calculate correlation coefficients
    n_corr = np.corrcoef(original_displacements['north'], reconstructed['north'])[0, 1]
    e_corr = np.corrcoef(original_displacements['east'], reconstructed['east'])[0, 1]
    u_corr = np.corrcoef(original_displacements['vertical'], reconstructed['vertical'])[0, 1]

    # Calculate explained variance
    n_var_explained = 1 - np.var(n_residuals) / np.var(original_displacements['north'])
    e_var_explained = 1 - np.var(e_residuals) / np.var(original_displacements['east'])
    u_var_explained = 1 - np.var(u_residuals) / np.var(original_displacements['vertical'])

    # Calculate signal amplitudes
    n_amplitude = np.std(original_displacements['north'])
    e_amplitude = np.std(original_displacements['east'])
    u_amplitude = np.std(original_displacements['vertical'])

    # Calculate signal-to-residual ratios
    n_snr = n_amplitude / np.std(n_residuals) if np.std(n_residuals) > 0 else float('inf')
    e_snr = e_amplitude / np.std(e_residuals) if np.std(e_residuals) > 0 else float('inf')
    u_snr = u_amplitude / np.std(u_residuals) if np.std(u_residuals) > 0 else float('inf')

    # Prepare validation results
    validation = {
        'residuals': {
            'north': n_residuals,
            'east': e_residuals,
            'up': u_residuals
        },
        'rms': {
            'north': n_rms,
            'east': e_rms,
            'up': u_rms,
            'total': total_rms
        },
        'correlation': {
            'north': n_corr,
            'east': e_corr,
            'up': u_corr
        },
        'variance_explained': {
            'north': n_var_explained,
            'east': e_var_explained,
            'up': u_var_explained
        },
        'signal_to_residual': {
            'north': n_snr,
            'east': e_snr,
            'up': u_snr
        },
        'reconstructed': reconstructed
    }

    # Print summary
    print(f"Validation Summary:")
    print(f"  RMS Errors: North={n_rms:.3e}m, East={e_rms:.3e}m, Up={u_rms:.3e}m, Total={total_rms:.3e}m")
    print(f"  Correlation: North={n_corr:.4f}, East={e_corr:.4f}, Up={u_corr:.4f}")
    print(f"  Variance explained: North={n_var_explained:.2%}, East={e_var_explained:.2%}, Up={u_var_explained:.2%}")
    print(f"  Signal-to-Residual Ratio: North={n_snr:.2f}, East={e_snr:.2f}, Up={u_snr:.2f}")

    return validation


def analyze_coefficient_spectrum(coeffs, max_degree=None):
    """
    Perform spectral analysis on the spherical harmonic coefficients.
    """
    # Extract coefficient arrays
    if 'load_coefficients' in coeffs:
        c_array = coeffs['load_coefficients'].coeffs
    else:
        c_array = coeffs.coeffs

    if max_degree is None:
        max_degree = c_array.shape[1] - 1

    # Calculate degree variance (power per degree)
    degrees = np.arange(1, max_degree + 1)
    power = np.zeros(max_degree)

    for n in range(1, max_degree + 1):
        power_sum = 0
        for m in range(n + 1):
            power_sum += c_array[0, n, m] ** 2  # C terms
            if m > 0:
                power_sum += c_array[1, n, m] ** 2  # S terms

        power[n - 1] = power_sum

    # Calculate cumulative power
    cumulative_power = np.cumsum(power)
    total_power = cumulative_power[-1]
    normalized_cumulative = cumulative_power / total_power if total_power > 0 else np.zeros_like(cumulative_power)

    # Calculate cutoff degree (where 99% of power is captured)
    cutoff_99 = np.argmax(normalized_cumulative >= 0.99) + 1 if any(normalized_cumulative >= 0.99) else max_degree
    cutoff_95 = np.argmax(normalized_cumulative >= 0.95) + 1 if any(normalized_cumulative >= 0.95) else max_degree
    cutoff_90 = np.argmax(normalized_cumulative >= 0.90) + 1 if any(normalized_cumulative >= 0.90) else max_degree

    # Create spectral plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot degree variance
    ax1.semilogy(degrees, power, 'b-o')
    ax1.set_xlabel('Degree (n)')
    ax1.set_ylabel('Power')
    ax1.set_title('Spherical Harmonic Power Spectrum')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)

    # Plot cumulative power
    ax2.plot(degrees, normalized_cumulative * 100, 'r-o')
    ax2.axhline(y=90, color='g', linestyle='--', alpha=0.7, label='90%')
    ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95%')
    ax2.axhline(y=99, color='purple', linestyle='--', alpha=0.7, label='99%')
    ax2.axvline(x=cutoff_90, color='g', linestyle=':', alpha=0.7)
    ax2.axvline(x=cutoff_95, color='orange', linestyle=':', alpha=0.7)
    ax2.axvline(x=cutoff_99, color='purple', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Degree (n)')
    ax2.set_ylabel('Cumulative Power (%)')
    ax2.set_title('Cumulative Power Spectrum')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()

    # Return analysis results
    return {
        'degrees': degrees,
        'power': power,
        'cumulative_power': normalized_cumulative,
        'cutoff_90': cutoff_90,
        'cutoff_95': cutoff_95,
        'cutoff_99': cutoff_99,
        'figure': fig
    }

def plot_residual_map(validation_results, save_path=None, title=None):
    """
    Plot spatial distribution of residuals on a map using Cartopy.

    Parameters:
    -----------
    validation_results : dict
        Validation results from validate_coefficient_solution
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Title for the figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Extract data
    lats = validation_results['reconstructed']['lat']
    lons = validation_results['reconstructed']['lon']
    n_res = validation_results['residuals']['north']
    e_res = validation_results['residuals']['east']
    u_res = validation_results['residuals']['up']

    # Calculate horizontal and total residuals
    h_res = np.sqrt(n_res ** 2 + e_res ** 2)
    total_res = np.sqrt(n_res ** 2 + e_res ** 2 + u_res ** 2)

    # Create figure with four maps (smaller size)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             subplot_kw={'projection': ccrs.Robinson()})

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Data to plot
    residuals = [n_res, e_res, u_res, total_res]
    titles = ['North Residuals', 'East Residuals', 'Up Residuals', 'Total Residuals']

    # Set fixed limits for components and RMS
    vmin_components = -0.005
    vmax_components = 0.005
    vmax_total = 0.005

    for i, (res, subtitle) in enumerate(zip(residuals, titles)):
        ax = axes[i]

        # Add map features
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        # Set color scaling based on plot type
        if i < 3:  # North, East, Up components
            cmap = cm.RdBu_r
            vmin = vmin_components
            vmax = vmax_components
        else:  # Total residual (always positive)
            cmap = 'RdBu_r'
            vmin = 0
            vmax = vmax_total

        # Plot residuals
        sc = ax.scatter(lons, lats, c=res, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=20, alpha=0.7, edgecolor='k', linewidth=0.2,
                        transform=ccrs.PlateCarree())

        # Create horizontal colorbar below the map
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.3, axes_class=plt.Axes)
        cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
        cbar.set_label('Residual (m)')

        # Add RMS value to title
        rms = np.sqrt(np.mean(res ** 2))
        ax.set_title(f'{subtitle} (RMS: {rms:.2e} m)')

    # Add main title if provided
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    else:
        fig.suptitle('Spatial Distribution of Forward Modeling Residuals', fontsize=14, y=0.98)

    # Adjust layout with more space for colorbars
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.35,wspace=0.0)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual map saved to {save_path}")

    return fig

def plot_displacement_comparison_map(original_displacements, reconstructed_displacements,
                                     component='vertical', save_path=None, title=None):
    """
    Plot original and reconstructed displacements side by side on maps using Cartopy.

    Parameters:
    -----------
    original_displacements : dict
        Original displacement data
    reconstructed_displacements : dict
        Reconstructed displacement data
    component : str, optional
        Displacement component to plot ('vertical', 'north', 'east', or 'total')
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Title for the figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Extract data
    lats = original_displacements['lat']
    lons = original_displacements['lon']

    # Determine which component to plot
    if component == 'vertical':
        orig_data = original_displacements['vertical']
        recon_data = reconstructed_displacements['vertical']
        comp_title = 'Vertical Displacement'
    elif component == 'north':
        orig_data = original_displacements['north']
        recon_data = reconstructed_displacements['north']
        comp_title = 'North Displacement'
    elif component == 'east':
        orig_data = original_displacements['east']
        recon_data = reconstructed_displacements['east']
        comp_title = 'East Displacement'
    elif component == 'total':
        # Calculate total magnitude
        orig_data = np.sqrt(
            original_displacements['north'] ** 2 +
            original_displacements['east'] ** 2 +
            original_displacements['vertical'] ** 2
        )
        recon_data = np.sqrt(
            reconstructed_displacements['north'] ** 2 +
            reconstructed_displacements['east'] ** 2 +
            reconstructed_displacements['vertical'] ** 2
        )
        comp_title = 'Total Displacement'
    else:
        raise ValueError(f"Unknown component: {component}")

    # Calculate residuals
    residuals = orig_data - recon_data

    # Create figure with three maps (smaller size)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4),
                             subplot_kw={'projection': ccrs.Robinson()})

    # Data to plot
    data_sets = [orig_data, recon_data, residuals]
    subtitles = ['Original', 'Reconstructed', 'Residual']

    # Set fixed limits based on component
    if component in ['north', 'east', 'vertical']:
        vmin = -0.02
        vmax = 0.02
        res_vmin = -0.02
        res_vmax = 0.02
    else:  # total displacement is always positive
        vmin = 0
        vmax = 0.02
        res_vmin = -0.02
        res_vmax = 0.02

    for i, (data, subtitle) in enumerate(zip(data_sets, subtitles)):
        ax = axes[i]

        # Add map features
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        # Choose colormap and scale
        if i < 2:  # Original and reconstructed
            cmap = cm.RdBu_r#cm.viridis
            v_min, v_max = vmin, vmax
        else:  # Residuals
            cmap = cm.RdBu_r
            v_min, v_max = res_vmin, res_vmax

        # Plot data
        sc = ax.scatter(lons, lats, c=data, cmap=cmap, vmin=v_min, vmax=v_max,
                        s=20, alpha=0.7, edgecolor='k', linewidth=0.2,
                        transform=ccrs.PlateCarree())

        # Create horizontal colorbar below the map
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.3, axes_class=plt.Axes)
        cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
        cbar.set_label('Displacement (m)')

        # Add RMS to residual plot
        if i == 2:
            rms = np.sqrt(np.mean(data ** 2))
            subtitle = f'{subtitle} (RMS: {rms:.2e} m)'

        ax.set_title(subtitle)

    # Add main title
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    else:
        fig.suptitle(f'{comp_title} Comparison', fontsize=14, y=0.98)

    # Adjust layout with more space for colorbars
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Displacement comparison map saved to {save_path}")

    return fig

def plot_displacement_components(displacements, save_path=None, title=None):
    """
    Plot north, east, and vertical displacement components side by side on maps using Cartopy.

    Parameters:
    -----------
    displacements : dict
        Displacement data containing 'lat', 'lon', 'north', 'east', and 'vertical' components
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Title for the figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Extract data
    lats = displacements['lat']
    lons = displacements['lon']

    # Create figure with three maps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4),
                             subplot_kw={'projection': ccrs.Robinson()})

    # Data to plot
    components = ['north', 'east', 'vertical']
    component_titles = ['North Displacement', 'East Displacement', f'Vertical Displacement | {len(lats)} POINTS']

    # Set consistent limits for all components
    vmin = -0.02
    vmax = 0.02

    for i, (component, comp_title) in enumerate(zip(components, component_titles)):
        ax = axes[i]
        data = displacements[component]

        # Add map features
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        # Choose colormap
        cmap = 'RdBu_r'#cm.viridis

        # Plot data
        sc = ax.scatter(lons, lats, c=data, cmap=cmap, vmin=vmin, vmax=vmax,
                      s=20, alpha=0.7, edgecolor='k', linewidth=0.2,
                      transform=ccrs.PlateCarree())

        # Create horizontal colorbar below the map
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.3, axes_class=plt.Axes)
        cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
        cbar.set_label('Displacement (m)')

        ax.set_title(comp_title)

    # Add main title
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    else:
        fig.suptitle('Displacement Components', fontsize=14, y=0.98)

    # Adjust layout with more space for colorbars
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Displacement components map saved to {save_path}")

    return fig

def plot_vector_displacement_map(displacements, scale=1.0, save_path=None, title=None):
    """
    Plot horizontal displacement vectors on a map using Cartopy.

    Parameters:
    -----------
    displacements : dict
        Displacement data with 'north', 'east', 'lat', 'lon' keys
    scale : float, optional
        Scaling factor for vector arrows
    save_path : str, optional
        Path to save the figure
    title : str, optional
        Title for the figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np

    # Extract data
    lats = displacements['lat']
    lons = displacements['lon']
    north = displacements['north']
    east = displacements['east']

    # Optional: vertical component as color
    if 'vertical' in displacements:
        vertical = displacements['vertical']
    else:
        vertical = None

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.Robinson()})

    # Add map features
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    # Calculate magnitude for scaling
    magnitude = np.sqrt(north ** 2 + east ** 2)
    max_mag = np.max(magnitude)

    # Scale vectors for visibility
    u = east / max_mag * scale
    v = north / max_mag * scale

    # Plot vectors
    if vertical is not None:
        # Use vertical component for color
        quiv = ax.quiver(lons, lats, u, v, vertical,
                         transform=ccrs.PlateCarree(),
                         cmap='RdBu_r', scale=1.0, scale_units='inches',
                         width=0.002, headwidth=5, headlength=5)

        # Add colorbar
        cbar = plt.colorbar(quiv, ax=ax, pad=0.01)
        cbar.set_label('Vertical Displacement (m)')
    else:
        # Plain vectors
        quiv = ax.quiver(lons, lats, u, v,
                         transform=ccrs.PlateCarree(),
                         scale=1.0, scale_units='inches',
                         width=0.002, headwidth=5, headlength=5, color='blue')

    # Add vector scale reference
    ref_mag = np.mean(magnitude)
    plt.quiverkey(quiv, 0.9, 0.9, ref_mag, f'{ref_mag:.2e} m',
                  labelpos='E', coordinates='figure')

    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Horizontal Displacement Vectors')

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Vector displacement map saved to {save_path}")

    return fig

def analyze_temporal_evolution(base_dir, start_date=None, end_date=None, coefficient_type='load'):
    """
    Analyze the temporal evolution of coefficient errors.

    Parameters:
    -----------
    base_dir : str
        Base directory containing date-specific subdirectories with results
    start_date : datetime, optional
        Start date for analysis
    end_date : datetime, optional
        End date for analysis
    coefficient_type : str
        Type of coefficients to analyze ('load' or 'potential')

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with temporal analysis plots
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    import glob
    import re

    # Find all date directories
    date_dirs = sorted(glob.glob(os.path.join(base_dir, '[0-9]' * 8)))

    if not date_dirs:
        print(f"No date directories found in {base_dir}")
        return None

    # Filter by date range if specified
    filtered_dirs = []
    for date_dir in date_dirs:
        date_str = os.path.basename(date_dir)
        if re.match(r'\d{8}', date_str):  # Make sure it's a date directory
            try:
                date = datetime.datetime.strptime(date_str, '%Y%m%d')

                if start_date and date < start_date:
                    continue

                if end_date and date > end_date:
                    continue

                filtered_dirs.append((date, date_dir))
            except ValueError:
                continue

    if not filtered_dirs:
        print("No directories match the date range criteria.")
        return None

    # Sort by date
    filtered_dirs.sort()

    # Prepare data containers
    dates = []
    degree1_errors = []
    degree2_errors = []
    mean_snr = []
    residuals = []

    for date, date_dir in filtered_dirs:
        # Find coefficient files
        coeff_files = glob.glob(os.path.join(date_dir, f"*{coefficient_type}*.txt"))
        summary_files = glob.glob(os.path.join(date_dir, "*summary.txt"))

        # Skip if no coefficient files found
        if not coeff_files:
            print(f"No {coefficient_type} coefficient files found in {date_dir}")
            continue

        # Find coefficient file with errors
        error_file = None
        for file in coeff_files:
            with open(file, 'r') as f:
                header = f.readline() + f.readline()
                if 'sigma' in header:
                    error_file = file
                    break

        # Skip if no error information available
        if not error_file:
            print(f"No error information found in {date_dir}")
            continue

        # Process the error file
        try:
            data = np.loadtxt(error_file, skiprows=2)

            # Extract degree 1 and 2 errors
            degree1_mask = data[:, 0] == 1
            degree2_mask = data[:, 0] == 2

            if np.any(degree1_mask):
                degree1_error = np.mean(data[degree1_mask, 4:6])  # columns 4-5 are sigma_C and sigma_S
                degree1_errors.append(degree1_error)
            else:
                degree1_errors.append(np.nan)

            if np.any(degree2_mask):
                degree2_error = np.mean(data[degree2_mask, 4:6])
                degree2_errors.append(degree2_error)
            else:
                degree2_errors.append(np.nan)

            # Calculate mean SNR across all coefficients
            if data.shape[1] > 6:  # Make sure SNR columns exist
                mean_snr.append(np.nanmean(data[:, 6:8]))  # columns 6-7 are SNR_C and SNR_S
            else:
                mean_snr.append(np.nan)

            # Add the date
            dates.append(date)

            # Get residual from summary file if available
            if summary_files:
                with open(summary_files[0], 'r') as f:
                    content = f.read()
                    # Extract RMS of residuals
                    match = re.search(r'RMS of residuals: ([\d\.e\-+]+)', content)
                    if match:
                        residuals.append(float(match.group(1)))
                    else:
                        residuals.append(np.nan)
            else:
                residuals.append(np.nan)

        except Exception as e:
            print(f"Error processing {date_dir}: {e}")

    if not dates:
        print("No valid data found for analysis.")
        return None

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Plot errors
    axes[0].plot(dates, degree1_errors, 'o-', label='Degree 1')
    axes[0].plot(dates, degree2_errors, 's-', label='Degree 2')
    axes[0].set_ylabel('Average Coefficient Error')
    axes[0].set_title(f'Temporal Evolution of {coefficient_type.capitalize()} Coefficient Errors')
    axes[0].legend()
    axes[0].grid(True)

    # Plot SNR
    axes[1].plot(dates, mean_snr, 'o-', color='green')
    axes[1].set_ylabel('Average SNR')
    axes[1].set_title(f'Temporal Evolution of {coefficient_type.capitalize()} Coefficient SNR')
    axes[1].grid(True)

    # Plot residuals
    if any(not np.isnan(r) for r in residuals):
        axes[2].plot(dates, residuals, 'o-', color='red')
        axes[2].set_ylabel('RMS of Residuals (m)')
        axes[2].set_title('Temporal Evolution of Reconstruction Residuals')
        axes[2].grid(True)
    else:
        axes[2].set_visible(False)

    # Format x-axis
    axes[-1].xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(base_dir, f"{coefficient_type}_temporal_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Temporal analysis plot saved to {output_file}")

    return fig

def plot_station_weights(station_lats, station_lons, normalized_weights, station_ids,
                         save_path=None, weight_threshold=0.5):
    """
    Plot station weights on a world map.

    Parameters
    ----------
    station_lats : array-like
        Station latitudes
    station_lons : array-like
        Station longitudes
    normalized_weights : dict
        Dictionary of normalized weights from VCE
    station_ids : array-like
        Station IDs corresponding to the keys in normalized_weights
    save_path : str, optional
        Path to save the plot
    weight_threshold : float, optional
        Threshold to highlight stations with weights below this value
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np

    # Create figure with map projection
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.Robinson())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Extract weights for each station
    weights = []
    for i, station_id in enumerate(station_ids):
        key = f'station_{station_id}'
        if key in normalized_weights:
            weights.append(normalized_weights[key])
        else:
            weights.append(1.0)  # Default weight if not found

    weights = np.array(weights)

    # Create colormap
    cmap = cm.RdYlGn  # Red-Yellow-Green colormap (red=low weights, green=high weights)

    # Plot all stations
    sc = ax.scatter(station_lons, station_lats, c=weights,
                    cmap=cmap, transform=ccrs.PlateCarree(),
                    s=60, alpha=0.8, vmin=0, vmax=1)

    # Highlight stations with low weights
    low_weight_mask = weights < weight_threshold
    if np.any(low_weight_mask):
        ax.scatter(np.array(station_lons)[low_weight_mask],
                   np.array(station_lats)[low_weight_mask],
                   c='none', edgecolor='red', s=120, linewidth=2,
                   transform=ccrs.PlateCarree())

        # Add station labels for low-weight stations
        for i, (lon, lat) in enumerate(zip(np.array(station_lons)[low_weight_mask],
                                           np.array(station_lats)[low_weight_mask])):
            station_id = np.array(station_ids)[low_weight_mask][i]
            weight = weights[low_weight_mask][i]
            ax.text(lon, lat, f"{station_id} ({weight:.2f})",
                    transform=ccrs.PlateCarree(),
                    fontsize=8, ha='right', va='bottom')

    # Add colorbar
    cbar = plt.colorbar(sc, orientation='horizontal', pad=0.05, aspect=30)
    cbar.set_label("Station Weight")

    # Add title and grid
    plt.title(f"Station Weights from Variance Component Estimation\n" +
              f"{np.sum(low_weight_mask)}/{len(weights)} stations below threshold ({weight_threshold})")
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()