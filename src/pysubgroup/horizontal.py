import datetime
import math
import pysubgroup as ps	
import cupy as cp
import cudf

#create the search space on GPU, creating every selector's cover array
#@ensure_df_type_set
class gpu_search_space:
    def __init__(self, data, target_attribute, target_value, nbins=5, ignore=[]):
        self.data = data
        
        self.instances = data.shape[0]
        a = datetime.datetime.now()
        self.sels = cudf.DataFrame({'id': 0,
                                    'attribute': target_attribute,
                                    'low' : 0,
                                    'high': 0,
                                    'type': 'target'
                                    })
        self.target_repr = self.compute_target_repr(target_attribute, target_value)
        self.positives = cp.count_nonzero(self.target_repr)
        self.sel_id = 1
        self.create_selectors_on_GPU(nbins, ignore)
        self.sels = self.sels.set_index('id', drop=False)
        self.stats = statistics_GPU(self)
        self.sel_stats = self.stats.compute_stats_sels()
        del self.data
        
        
    def compute_target_repr(self, target_attribute, target_value):
        return cp.from_dlpack((self.data[target_attribute] == target_value).to_dlpack())
    
    def create_selectors_on_GPU(self, nbins=5, ignore=[]):
        self.create_numeric_gpu(nbins, ignore)
        self.create_categorical_gpu(ignore)
        
    
    def create_numeric_gpu(self, nbins, ignore):
        indices = cp.arange(0, self.data.shape[0], self.data.shape[0]//nbins+1)
        selectors = [self.sels]
        representations = [self.target_repr.reshape(-1,1)]
        for attribute in [
        col for col in self.data.select_dtypes(include=["number"]) if col not in ignore]:
            uniques = cp.unique(self.data[attribute])
            #single value column, no selectors needed
            if len(uniques) == 1:
                continue
            #create categorical selector
            elif len(uniques)<=nbins:
                selectors.append(cudf.DataFrame({"id" : cp.arange(self.sel_id, self.sel_id + len(uniques)),
                                              "attribute" : attribute,
                                              "low" : uniques,
                                              "high": 0,
                                              "type" : "categorical"}))
                data = cp.from_dlpack(self.data[attribute].to_dlpack()).reshape(-1,1)
                uniques = uniques.reshape(1,-1)
                representations.append(data == uniques)
                
                self.sel_id += len(uniques)
            #create interval selectors
            else:
                sorted_data = cp.sort(cp.from_dlpack((self.data[attribute]).to_dlpack()))
                maximum = sorted_data[-1]+1
                values = cp.unique(sorted_data.take(indices))
                
                if values.dtype == cp.int_:
                    values = cp.append(values, maximum)
                else:
                    values = cp.append(values, cp.inf)
                if len(values) > 1:
                    selectors.append(cudf.DataFrame({"id" : cp.arange(self.sel_id, self.sel_id + len(values)-1),
                                                     "attribute" : attribute,
                                                     "low" : values[:-1],
                                                     "high": values[1:],
                                                     "type" : "interval"}))
                    self.sel_id += len(values)-1
                    data = cp.from_dlpack(self.data[attribute].to_dlpack()).reshape(-1,1)
                    values = values.reshape(1,-1)
                    representations.append(cp.logical_and(data >= values[:,:-1], data < values[:,1:]))
                
        self.sels = cudf.concat(selectors)
        self.reps = cp.swapaxes(cp.hstack(representations), 0, 1)
        
    #TODO: need to replace categories with ints for this one    
    def create_categorical_gpu(self, ignore):
        pass

        
    
                
class statistics_GPU:
    statistic_types = (
        "size_sg",
        "size_dataset",
        "positives_sg",
        "positives_dataset",
        "size_complement",
        "relative_size_sg",
        "relative_size_complement",
        "coverage_sg",
        "coverage_complement",
        "target_share_sg",
        "target_share_complement",
        "target_share_dataset",
        "lift",
    )
    def __init__(self, search_space):
        self.search_space = search_space
        self.constant_stats = cudf.DataFrame()
        self.compute_constant()
        
    def compute_constant(self):
        size = self.search_space.data.shape[0]
        pos = self.search_space.positives
        tshare = pos/size
        
        self.constant_stats['size_sg'] = [size]
        self.constant_stats['size_dataset'] = [size]
        self.constant_stats['positives_sg'] = [pos]
        self.constant_stats['positives_dataset'] = [pos]
        self.constant_stats['size_complement'] = [0]
        self.constant_stats['relative_size_sg'] = [1]
        self.constant_stats['coverage_sg']= [1]
        self.constant_stats['relative_size_complement'] = [0]
        self.constant_stats['coverage_complement']= [0]
        self.constant_stats['target_share_sg'] = [tshare]
        self.constant_stats['target_share_dataset'] = [tshare]
        self.constant_stats['target_share_complement'] = [float('nan')]
        self.constant_stats['lift'] = [1]
    
    def compute_stats(self, counts, positives):
        statistics = cudf.DataFrame({'size_sg' : counts, 'positives_sg': positives})
        statistics['size_complement'] = (-statistics['size_sg']).add(self.constant_stats['size_dataset'].iloc[0])
        statistics['relative_size_sg'] = statistics['size_sg'].truediv(self.constant_stats['size_dataset'].iloc[0])
        statistics['relative_size_complement'] = (-statistics['relative_size_sg']).add(1)
        statistics['coverage_sg'] = statistics['positives_sg'].truediv(self.constant_stats['positives_dataset'].iloc[0])
        statistics['coverage_complement'] = 1 - statistics['coverage_sg']
        statistics['target_share_sg'] = statistics['positives_sg'] / statistics['size_sg']
        statistics['target_share_complement'] = ((- statistics['positives_sg']).add(self.constant_stats['positives_dataset'].iloc[0])) /statistics['size_complement']
        statistics['lift'] = statistics['target_share_sg'].truediv(self.constant_stats['target_share_dataset'].iloc[0])
        return statistics
    
    def compute_quality(self, statistics, a):
        return (statistics['relative_size_sg'].pow(a) * (statistics['target_share_sg'].add(-self.constant_stats['target_share_dataset'].iloc[0])))
    
    def compute_optimistic(self, statistics, a):
        return statistics['relative_size_sg'].pow(a).multiply(1-self.constant_stats['target_share_dataset'].iloc[0])
    
    #TODO: delete this failed experiment
    #subgroups should alredy be in int format  
    def compute_stats_subgroups(self, subgroups, batch_size = 0):
        
        nr_cols = self.search_space.representations.shape[1]
        nr_ins = self.search_space.representations.shape[0]
        nr_sgs = subgroups.shape[0]
        
        if batch_size == 0:
            batch_size = set_optimal_batch_size(nr_cols, nr_sgs)
        
        arr_sg = ~subgroups
        arr_pos = arr_sg.copy()
        arr_pos[:,0] = cp.bitwise_and(arr_pos[:,0],cp.uint64(0xFFFFFFFFFFFFFFFE))
        arr_sg=arr_sg.reshape(1, nr_sgs, nr_cols)
        arr_pos=arr_pos.reshape(1, nr_sgs, nr_cols)
        cnt = cp.zeros(shape = nr_sgs, dtype = int)
        pos = cp.zeros(shape = nr_sgs, dtype = int)
        
        #divide representations of selectors into (vertical) batches, OR with negative of selectors subgroup holds: If a selector is not in it, set data to True, otherwise keep its value. Then see if all values are true -> if yes, instance is in sg, otherwise it's not. Repeat for positives by modifying first column.
        for i in range(0,nr_ins,batch_size):
            print(i)
            batch_arr = self.search_space.representations[i:i+batch_size,:]
            batch_arr=batch_arr.reshape(-1, 1, nr_cols)
            
            
            mask_cnt = cp.bitwise_or(batch_arr, arr_sg) == cp.uint64(0xFFFFFFFFFFFFFFFF)
            mask_pos = cp.bitwise_or(batch_arr, arr_pos) == cp.uint64(0xFFFFFFFFFFFFFFFF)
            
            batch_cnt = cp.sum(cp.sum(mask_cnt, axis=2) == nr_cols, axis=0)
            batch_pos = cp.sum(cp.sum(mask_pos, axis=2) == nr_cols, axis=0)
            
            cnt += batch_cnt
            pos += batch_pos
        
        statistics = self.compute_stats(cnt, pos)
        return statistics
    
    def sel_conjunction_quality(self, sels, a):
        cover_arr = cp.all(self.search_space.reps[sels], axis = 0)
        cnt = cp.count_nonzero(cover_arr)
        pos = cp.count_nonzero(cover_arr[self.search_space.target_repr])
        stats = self.compute_stats(cnt, pos)
        return self.compute_quality(stats, a).iloc[0]
    
    def compute_stats_sels(self):
        reps = self.search_space.reps
        cnt = cp.count_nonzero(reps, axis=1)
        pos = cp.count_nonzero(reps & reps[0], axis=1)
        statistics = self.compute_stats(cnt, pos)
        return statistics
    

    
    def add_sel(self, old_arr, sel, a):
        new_arr = (old_arr & self.search_space.reps[:,sel] )
        cnt = cp.count_nonzero(new_arr)
        pos = cp.count_nonzero(new_arr[self.search_space.target_repr])
        stats = self.compute_stats(cnt, pos)
        return self.compute_quality(stats, a).iloc[0], new_arr
    
    def add_sels(self, old_arr, sels, a):
        new_arrs = (old_arr & self.search_space.reps[sels])
        cnts = cp.sum(new_arrs, axis=1)
        poss = cp.sum(new_arrs & self.search_space.reps[0], axis=1)
        stats = cudf.DataFrame()
        stats['size_sg'] = cnts
        stats['positives_sg'] = poss
        stats['relative_size_sg'] = stats['size_sg'].truediv(self.constant_stats['size_dataset'].iloc[0])
        stats['target_share_sg'] = stats['positives_sg'] / stats['size_sg']
        
        q = self.compute_quality(stats, a)
        o = self.compute_optimistic(stats, a)
        return q.to_arrow().to_pylist(), o.to_arrow().to_pylist(), new_arrs
    
    def get_cover_arr_sels(self, sels):
        cover_arr = cp.all(self.search_space.reps[sels], axis = 0)
        return cover_arr
        
        
        
 
    
#TODO: don't need these anymore
#convert boolean dataframe to int64 cupy array   
def bool_to_int(bool_array):
    #bool_array = cp.array(df.values)
    num_int64 = (bool_array.shape[1] + 63) // 64
    result = cp.zeros(shape=(bool_array.shape[0], num_int64), dtype=cp.uint64)
    for i in range(num_int64):
        start_col = i * 64
        end_col = min((i+1) * 64, bool_array.shape[1])
        chunk = bool_array[:,start_col:end_col]
        bit_mask = 2 ** cp.arange(chunk.shape[1], dtype=cp.uint64)
        result[:, i] = (chunk * bit_mask).sum(axis=1, dtype=cp.uint64)
    return result
    
def set_optimal_batch_size(nr_cols, nr_sgs, safety = 1):
    available_mem = cp.cuda.runtime.memGetInfo()[0]
    slice_size = nr_cols * nr_sgs * 16
    return math.floor((available_mem*safety) // slice_size)


#testing area
if __name__ == '__main__':
    folder = '/home/alexpuff/datasets'
    spam_csv = '/synth_spam.csv'
    iris_csv = '/synth_iris.csv'
    darwin_csv = '/synth_darwin.csv'
    spam_ignore=['Class', 'word_freq_email']
    iris_ignore=['class']
    darwin_ignore=['ID', 'class']
    #spam_target=ps.BinaryTarget('Class', 1)
    #iris_target=ps.BinaryTarget('class', 'Iris-virginica')
    #darwin_target=ps.BinaryTarget('class', 'H')
    
    if True:
        start = datetime.datetime.now()
        df = cudf.read_csv(folder+spam_csv,sep="\t", header=0, nrows = 5000000)
        print(f"Data loaded: {datetime.datetime.now() - start}")
        sp = gpu_search_space(df, 'Class', 1, 2, spam_ignore)
        a = datetime.datetime.now()
        task = ps.gpu_task(sp, 1, depth=3, result_set_size=10)
        dfs = ps.gpu_dfs(task)
        res = dfs.execute(caching=False)
        print(res.iloc[0])
        print(f"Execution time: {datetime.datetime.now() - a}")
    #df = cudf.DataFrame({'a' : [1,2,3], 'b' : [4,5,6]})
    #arr = cp.array([1,2])
    #print(df['a'].to_cupy().reshape(-1,1) == arr.reshape(1,-1))
    
        
    
    
    
    #NUMBA_CUDA_DRIVER="/usr/lib/wsl/lib/libcuda.so.1"
        
    
