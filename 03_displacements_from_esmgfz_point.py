import pandas as pd
from glob import glob
import os

model = 'O'
frame = 'cf'

out_dir_time = rf'DATA/DISPLACEMENTS/ESMGFZ_{model}_{frame}_IGSNET/TIME/'
os.makedirs(out_dir_time,exist_ok=True)

files = glob(rf'EXT/ESMGFZLOADING/CODE/*_{model}_{frame}.PKL')

files_df = []
for x in files:
    code = os.path.basename(x)
    code = code.split('_')[0]
    x_df = pd.read_pickle(x)

    x_df['CODE'] = code
    files_df.append(x_df)

results_df = pd.concat(files_df,axis=0).reset_index()

results_df = results_df.rename({'R': 'dU', 'NS': 'dN', 'EW': 'dE', 'datetime':'EPOCH','index':'EPOCH'}, axis=1)

dftime = results_df.groupby('EPOCH')

for time, df in dftime:
    str_time = time.strftime('%Y%m%d')
    df = df.set_index(['EPOCH','CODE'])
    df.sort_index(level='CODE').to_pickle(os.path.join(out_dir_time,f'ESMGFZ_{str_time}_{model}_{frame}_DISP.PKL'))


