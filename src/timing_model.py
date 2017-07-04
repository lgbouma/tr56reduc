import numpy as np
import pandas as pd

df = pd.read_csv('../doc/OGLE_TR_56b/Adams_et_al_2011_table6.txt',
        delimiter='\t')

df['Midtime err (BJD_TDB)'] = [s.split(' +or- ')[1] for s in df['Midtime (BJD_TDB)']]
df['Midtime (BJD_TDB)'] = [s.split(' +or- ')[0] for s in df['Midtime (BJD_TDB)']]

df['O-C err (s)'] = [s.split(' +or- ')[1] for s in df['O-C (s)']]
df['O-C (s)'] = [s.split(' +or- ')[0] for s in df['O-C (s)']]

df = df.drop('Unnamed: 6', axis=1)

# Or just follow Eq 1:

N = np.arange(0,3300,1)

T_C = 2453936.60070 + 1.21191096 * N

epoch_T_C = np.array((N, T_C)).T
