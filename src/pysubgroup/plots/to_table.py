
import numpy as np
import pandas as pd
from itertools import product


dir = '/home/alexpuff/runtimes/FullTest/'

def csv_to_table(path, avg_over, split_on, output, sum_up):
    df = pd.read_csv(path)
    
    if avg_over:
        df = df.groupby(avg_over + split_on, as_index=False).agg({
            col: 'mean' if pd.api.types.is_numeric_dtype(df[col]) else 'first'
            for col in output
        })
    split_values = {col: sorted(df[col].unique()) for col in split_on}
    table = []
    
    first = True
    cnt = 0
    for avged in sorted(df[avg_over[0]].unique()):
        
        col = df[df[avg_over[0]] == avged]
        s = [str(avged), ' & ']
        for p in product(*split_values.values()):
            
            mask = pd.Series(True, index=df.index)
            for col_nr, val in enumerate(p):
                mask &= df[split_on[col_nr]] == val
            if mask.any():
                
                if first:
                    cnt += 1
                    print(p)
                df_here = df[mask]
                col = df_here[df_here[avg_over[0]] == avged]
                if sum_up:
                    try:
                        val = 0
                        for o in output:
                            val += col[o].values[0]
                        s.append(str(round(float(val), 3)))
                    except:
                        s.append('OOM')
                    s.append(' & ')
                else:
                    for o in output:
                        try:
                            s.append(str(round(float(col[o].values[0]), 3)))
                        except:
                            s.append('OOM')
                        s.append(' & ')
                
        s = ''.join(s)
        s = s[:-2]
        s += '\\\\'
        print(s)
        first = False
    
    tabular = '|l|'
    for i in range(cnt):
        tabular += 'r|'
    print(tabular)
if __name__ == '__main__':
    config = {
        'path': dir + 'memory/memdepth.csv',
        'avg_over' : ['Depth'],
        'split_on' : ['Device', 'GPU'],
        'output' : ['RAM', 'VRAM'],
        'sum_up' : True
    }
    csv_to_table(**config)