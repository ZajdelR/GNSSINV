import gcc_analysis_tools as gcc
from toolbox_gfc_analysis import *

suffix = '_REG'

TN14 = pd.read_pickle('EXT/TN-14_C20_C30').reset_index()

TN14_C20 = TN14[['MJD_start','C20_mean']].rename({'MJD_start':'date', 'C20_mean':'C'}, axis=1).set_index('date')
TN14_C20['C_error'] = 0

TN14_C30 = TN14[['MJD_start','C30_mean']].rename({'MJD_start':'date', 'C30_mean':'C'}, axis=1).set_index('date')
TN14_C30['C_error'] = 0

c20 = {}
kwargs = dict(coeff_type='potential', base_dir='OUTPUT', degree=2, order=0,
              start_date='2015-01-01',end_date='2020-01-01')

c20['TN14'] = TN14_C20

# for deg in [3,5,7,9]:
#     c20[f'IGS-{deg}'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D_WO-AOS', degmax=deg,
#                                              **kwargs)

c20['IGS-9'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D_WO-AOS', degmax=9,
                                         **kwargs)

c20['IGS-9-REG'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D_REG_WO-AOS', degmax=9,
                                         **kwargs)

# c20['IGS-AOS'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D_WO-AOS', degmax=7,
#                                          **kwargs)
#
# c20['IGS-RAW'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D', degmax=7,
#                                          **kwargs)
#
# c20['IGS-AOS_LIM'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D_LIM_WO-AOS', degmax=7,
#                                          **kwargs)
#
# c20['IGS-RAW_LIM'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D_LIM', degmax=7,
#                                          **kwargs)
#
# c20['ESM_GRID'] = create_coefficient_time_series(solution='ESMGFZ_LSDM_cf_GRIDS', degmax=7,
#                                          **kwargs)
#
# c20['ESM'] = create_coefficient_time_series(solution='ESMGFZ_H_cf_IGSNET', degmax=7,
#                                          **kwargs)
#
# c20['ESM_LIM'] = create_coefficient_time_series(solution='ESMGFZ_H_cf_IGSNET_LIM', degmax=7,
#                                          **kwargs)

c20 = gcc.filter_dataframes_dict(c20,'20150101','20200101')
c20 = gcc.resample_and_interpolate(c20,'1D')

c20_stats = plot_series_with_lombscargle(c20,'C',title='C20',units='',
                                         apply_lowpass_filter=apply_lowpass_filter,
                                         save_path=f'OUTPUT_PLOTS/20250409/C20_comparison{suffix}.png',
                                         y_offset=3e-10,
                                         width_cm=13,figsize=(1,1.5))

c20_signal = gcc.fit_and_provide_annual_semiannual_table_with_errors(c20,['C'])
plot_phasors(c20_signal,'C',width_cm=13,save_path=f'OUTPUT_PLOTS/20250409/C20_phasor{suffix}.png')

c30 = {}
kwargs = dict(coeff_type='potential', base_dir='OUTPUT', degree=3, order=0,
              start_date='2015-01-01',end_date='2020-01-01')

c30['TN14'] = TN14_C30

# for deg in [3,5,7,9]:
#     c30[f'IGS-{deg}'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D_WO-AOS', degmax=deg,
#                                              **kwargs)

c30['IGS-9'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D_WO-AOS', degmax=9,
                                         **kwargs)

c30['IGS-9-REG'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D_REG_WO-AOS', degmax=9,
                                         **kwargs)
#
# c30['IGS-RAW'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D', degmax=7,
#                                          **kwargs)
# c30['IGS-AOS'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D_WO-AOS', degmax=7,
#                                          **kwargs)
#
# c30['IGS-AOS_LIM'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D_LIM_WO-AOS', degmax=7,
#                                          **kwargs)
#
# c30['IGS-RAW_LIM'] = create_coefficient_time_series(solution='ITRF2020-IGS-RES_01D_LIM', degmax=7,
#                                          **kwargs)
#
# c30['ESM_GRID'] = create_coefficient_time_series(solution='ESMGFZ_LSDM_cf_GRIDS', degmax=7,
#                                          **kwargs)
#
# c30['ESM_IGS'] = create_coefficient_time_series(solution='ESMGFZ_H_cf_IGSNET', degmax=7,
#                                          **kwargs)
#
# c30['ESM_IGS_LIM'] = create_coefficient_time_series(solution='ESMGFZ_H_cf_IGSNET_LIM', degmax=7,
#                                          **kwargs)

c30 = gcc.filter_dataframes_dict(c30,'20150101','20200101')
c30 = gcc.resample_and_interpolate(c30,'1D')

c30_stats = plot_series_with_lombscargle(c30,'C',title='C30',units='',
                                         apply_lowpass_filter=apply_lowpass_filter,
                                         save_path=f'OUTPUT_PLOTS/20250409/C30_comparison{suffix}.png',
                                         y_offset=3e-10,
                                         width_cm=13,figsize=(1,1.5))

c30_signal = gcc.fit_and_provide_annual_semiannual_table_with_errors(c30,['C'])
plot_phasors(c30_signal,'C',width_cm=13,save_path=f'OUTPUT_PLOTS/20250409/C30_phasor{suffix}.png')
#
# c20_load = {}
# kwargs['coeff_type'] = 'load'
# c20_load['H_4'] = create_coefficient_time_series(solution='IGS1R03SNX_LOAD_CRD_CF_H_7D_HELMERT', degmax=4,
#                                          **kwargs)
#
# c20_load = gcc.filter_dataframes_dict(c20_load,'20000101','20200101')
# c20_load = gcc.resample_and_interpolate(c20_load,'1D')
#
# c20_load_stats = plot_series_with_lombscargle(c20_load,'C',title='C20',units='kg/m2',
#                                          apply_lowpass_filter=apply_lowpass_filter,
#                                          save_path=f'OUTPUT_WITH_ERRORS/C20_load_sample{suffix}.png',
#                                          y_offset=3e-10,
#                                          width_cm=13,figsize=(1,1.5))