import pandas as pd

import gcc_analysis_tools
from toolbox_gfc_analysis import *
from gcc_analysis_tools import load_data, geocenter_to_degree1,reverse_cf_cm

suffix = '_NOREG'

gcc = {}

kwargs = dict(date_start='2015-01-01', date_end='2020-01-01', base_dir='OUTPUT')
#
# gcc['IGS5'] = get_geocenter_motion('IGS1R03SNX_01D', degree=5, **kwargs)
# gcc['IGS8'] = get_geocenter_motion('IGS1R03SNX_01D', degree=8, **kwargs)
# gcc['IGS8_REG'] = get_geocenter_motion('IGS1R03SNX_01D_REG', degree=8, **kwargs)
# gcc['ESM'] = get_geocenter_motion('ESMGFZ_H_cf_IGSNET', degree=5, **kwargs)
# gcc['ESM_LIM'] = get_geocenter_motion('ESMGFZ_H_cf_IGSNET_LIM', degree=5, **kwargs)
#
gcc['SLR'] = geocenter_to_degree1(reverse_cf_cm(load_data(r'EXT/GCC/slr.gc')))

gcc['IGS-AOS'] = get_geocenter_motion(solution='ITRF2020-IGS-RES_01D_WO-AOS', degmax=7,
                                         **kwargs)

gcc['IGS-RAW'] = get_geocenter_motion(solution='ITRF2020-IGS-RES_01D', degmax=7,
                                         **kwargs)

gcc['IGS-AOS_LIM'] = get_geocenter_motion(solution='ITRF2020-IGS-RES_01D_LIM_WO-AOS', degmax=7,
                                         **kwargs)

gcc['IGS-RAW_LIM'] = get_geocenter_motion(solution='ITRF2020-IGS-RES_01D_LIM', degmax=7,
                                         **kwargs)

gcc['ESM_GRID'] = get_geocenter_motion(solution='ESMGFZ_LSDM_cf_GRIDS', degmax=7,
                                         **kwargs)

gcc['ESM_IGS'] = get_geocenter_motion(solution='ESMGFZ_H_cf_IGSNET', degmax=7,
                                         **kwargs)

gcc['ESM_IGS_LIM'] = get_geocenter_motion(solution='ESMGFZ_H_cf_IGSNET_LIM', degmax=7,
                                         **kwargs)

gcc['AHOS'] = geocenter_to_degree1(load_data(r'EXT/GCC/AHOS.gc'))
gcc['HYDL'] = geocenter_to_degree1(load_data(r'EXT/GCC/H.gc'))

gcc = gcc_analysis_tools.filter_dataframes_dict(gcc,'20150101','20200101')
gcc = gcc_analysis_tools.resample_and_interpolate(gcc,'1D')

gcc_stats = plot_series_with_lombscargle(gcc,'Z',title='Z',units='mm',
                                         apply_lowpass_filter=apply_lowpass_filter,
                                         save_path=f'OUTPUT_PLOTS/20250409/Z_comparison{suffix}.png',
                                         y_offset=20, periodogram_offset=0,
                                         width_cm=13,figsize=(1,1.5))

gcc_stats = plot_series_with_lombscargle(gcc,'C10',title='C10',units='kg/m2',
                                         apply_lowpass_filter=apply_lowpass_filter,
                                         save_path=f'OUTPUT_PLOTS/20250409/C10_comparison{suffix}.png',
                                         y_offset=100, periodogram_offset=0,
                                         width_cm=13,figsize=(1,1.5))

gcc_stats = plot_series_with_lombscargle(gcc,'Y',title='Y',units='mm',
                                         apply_lowpass_filter=apply_lowpass_filter,
                                         save_path=f'OUTPUT_PLOTS/20250409/Y_comparison{suffix}.png',
                                         y_offset=10, periodogram_offset=0,
                                         width_cm=13,figsize=(1,1.5))

gcc_stats = plot_series_with_lombscargle(gcc,'X',title='X',units='mm',
                                         apply_lowpass_filter=apply_lowpass_filter,
                                         save_path=f'OUTPUT_PLOTS/20250409/X_comparison{suffix}.png',
                                         y_offset=10, periodogram_offset=0,
                                         width_cm=13,figsize=(1,1.5))

gcc_stats_df = pd.DataFrame(gcc_stats)

annual_semiannual = gcc_analysis_tools.fit_and_provide_annual_semiannual_table_with_errors(gcc,['C11','S11','C10','X','Y','Z'])
# annual_semiannual = annual_semiannual.sort_values(['Component','Dataset'])

plot_phasors(annual_semiannual,'Z',width_cm=13,save_path=f'OUTPUT_PLOTS/20250409/Z_phasor{suffix}.png')
plot_phasors(annual_semiannual,'Y',width_cm=13,save_path=f'OUTPUT_PLOTS/20250409/X_phasor{suffix}.png')
plot_phasors(annual_semiannual,'X',width_cm=13,save_path=f'OUTPUT_PLOTS/20250409/Y_phasor{suffix}.png')
