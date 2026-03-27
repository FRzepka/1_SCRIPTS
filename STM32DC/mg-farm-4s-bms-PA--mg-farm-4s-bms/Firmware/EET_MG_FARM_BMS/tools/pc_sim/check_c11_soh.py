import pyarrow.parquet as pq
import numpy as np

pf = pq.ParquetFile(r'C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\0_Data\MGFarm_18650_FE\MGFarm_18650_C11.parquet')
tab = pf.read_row_group(0, columns=['SOH'])
soh = tab['SOH'].combine_chunks().to_numpy()

print('First 20 SOH values from C11 parquet:')
for i in range(min(20, len(soh))):
    print(f'{i}: {soh[i]:.6f}')

print(f'\nSOH range: {np.min(soh):.6f} - {np.max(soh):.6f}')
print(f'SOH mean: {np.mean(soh):.6f}')
