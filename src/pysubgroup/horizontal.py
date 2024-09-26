import datetime
import pandas as pd
import pysubgroup as ps	
import cupy as cp
import cudf
#from .dftype import DataFrameConfig, ensure_df_type_set

#create the search space on GPU, creating every selector's cover array
#@ensure_df_type_set
def create_selectors_on_GPU(data, nbins=5, ignore=[], caching = True):
    eq, interval = create_numeric_GPU(data.select_dtypes(include=["number"]), nbins, ignore, caching)
    categorical = create_categorical_GPU(data, ignore)
    if eq or categorical:
        equality = cudf.concat([eq, categorical], axis=0)
    else:
        equality = cudf.DataFrame()
    if caching:
        if equality:
            equality = compute_repr(equality)
        interval = compute_repr(interval)
    return equality, interval

def create_numeric_GPU(data, nbins, ignore, caching):
    indices = cp.arange(0, data.shape[0], data.shape[0]//nbins)
    print(indices)
    #initialize empty dataframe to hold all selectors
    equals=[]
    intervals =[]
    for attribute in [
        col for col in data if col not in ignore]:
        uniques = cp.unique(data[attribute])
        if len(uniques) == 1:
            #single value column, no selectors needed
            continue
        elif len(uniques) <= nbins:
            #add eq sel for every unique value
            if caching:
                #sel_data = cudf.concat([data[attribute].rename(None)]*(len(uniques)), axis=0, ignore_index=True)
                sel_data = cudf.DataFrame(cp.tile(data[attribute], (len(uniques),1)))
                sel_df = cudf.concat([cudf.DataFrame({'attribute_name': attribute, 'value': uniques}),sel_data],axis=1)
            else:
                sel_df = cudf.DataFrame({'attribute_name': attribute, 'value': uniques})
            equals.append(sel_df)
        else:
            #add intervals
            sorted_data = cp.sort(cp.from_dlpack((data[attribute]).to_dlpack()))
            maximum = sorted_data[-1]+1
            values = cp.unique(sorted_data.take(indices))
            #if dtype is int, no inf supported
            if values.dtype == cp.int_:
                values = cp.append(values, maximum)
            else:
                values = cp.append(values, cp.inf)
            #dataframe holding every selector's interval cutpoints and a copy of its data column
            if len(values) > 1: #Sometimes, values are so concentrated in big datasets that only one selector would be created
                if caching:
                    #sel_data = cudf.concat([data[attribute]]*(len(values)-1), axis=0, ignore_index=True)
                    sel_data = cudf.DataFrame(cp.tile(data[attribute], (len(values)-1,1)))
                    sel_df = cudf.concat([cudf.DataFrame({'attribute_name': attribute, 'left': values[:-1], 'right': values[1:]}),sel_data],axis=1)
                else:
                    sel_df = cudf.DataFrame({'attribute_name': attribute, 'left': values[:-1], 'right': values[1:]})
                intervals.append(sel_df)
    if equals:
        equal_selectors = cudf.concat(equals)
    else:
        equal_selectors = None
    if intervals:
        interval_selectors=cudf.concat(intervals)
    else:
        interval_selectors=None
    return equal_selectors, interval_selectors

#TODO: transform objects to int (and keep dict to swap back at the end)
def create_categorical_GPU(data, ignore):
    return None

def compute_repr(selectors):
    if 'left' in selectors.columns:
        low = cudf.concat([selectors['left'].rename(None)]*(selectors.shape[1]-3),axis=1)
        high = cudf.concat([selectors['right'].rename(None)]*(selectors.shape[1]-3),axis=1)
        return cudf.concat([selectors.iloc[:,:3], selectors.iloc[:,3:].ge(low) & selectors.iloc[:,3:].lt(high)],axis=1)
    else:
        vals = cudf.concat([selectors['value'].rename(None)]*(selectors.shape[1]-2),axis=1)
        return cudf.concat([selectors.iloc[:,:2],selectors.iloc[:,2:].eq(vals)],axis=1)
    
#TODO: Send result to original pysubgroup pipeline for display
def get_result():
    pass
#TODO: Specify target, special selector that gets its own dataframe etc
class GPU_Target:
    pass
#TODO: Equivalent to original task, holding data, search space and other relevant information
class GPU_Task:
    pass
#TODO: Quality measures
class GPU_Quality:
    pass
#TODO: BFS with no pruning whatsoever
class GPU_BFS_NoPrune:
    pass
#TODO: BFS with easy pruning: early exclude children of subgroups that are too small
class GPU_BFS:
    pass
#TODO: BFS with full apriori pruning
class GPU_Apriori:
    pass



#testing area
if __name__ == '__main__':
    folder = '/home/alexpuff/datasets'
    spam_csv = '/synth_spam.csv'
    iris_csv = '/synth_iris.csv'
    darwin_csv = '/synth_darwin.csv'
    spam_ignore=['Class', 'word_freq_email']
    iris_ignore=['class']
    darwin_ignore=['ID', 'class']
    df = cudf.read_csv(folder+iris_csv,sep="\t", header=0, nrows = 10000000)
    start=datetime.datetime.now()
    print(create_selectors_on_GPU(df, 5, iris_ignore, caching=False))
    end=datetime.datetime.now()
    print("Time taken:", end-start)
