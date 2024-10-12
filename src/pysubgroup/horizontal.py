import datetime
import numpy as np
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
                selectors.append(cudf.DataFrame({"id" : cp.arange(self.sel_id, self.sel_id + len(uniques), dtype=cp.uint16),
                                              "attribute" : attribute,
                                              "low" : uniques,
                                              "high": 0,
                                              "type" : "categorical"}))
                self.sel_id += len(uniques)
                data = cp.from_dlpack(self.data[attribute].to_dlpack()).reshape(-1,1)
                uniques = uniques.reshape(1,-1)
                representations.append(data == uniques)
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
                    selectors.append(cudf.DataFrame({"id" : cp.arange(self.sel_id, self.sel_id + len(values)-1, dtype = cp.uint16),
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
    
    def add_sels(self, parent, sels, a):
        old_arr = self.get_cover_arr_sels(parent[parent != 0])
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
        return q, o
    
    #every sg here should have same depth
    def compute_quality_optimistic(self, sgs, a):
        depth = cp.sum(sgs[0] != 0).item()
        reps = self.search_space.reps[sgs[:,0]]
        for d in range(1,depth):
            reps = reps & self.search_space.reps[sgs[:,d]]
        cnt = cp.sum(reps, axis=1)
        pos = cp.sum(reps & reps[0], axis=1)
        
        stats = cudf.DataFrame()
        stats['size_sg'] = cnt
        stats['positives_sg'] = pos
        stats['relative_size_sg'] = stats['size_sg'].truediv(self.constant_stats['size_dataset'].iloc[0])
        stats['target_share_sg'] = stats['positives_sg'] / stats['size_sg']
        
        q = self.compute_quality(stats, a)
        o = self.compute_optimistic(stats, a)
        return q, o
    
    def get_cover_arr_sels(self, sels):
        cover_arr = cp.all(self.search_space.reps[sels], axis = 0)
        return cover_arr


#testing area
def copy_df(df, num_copies):
    df_list = [df.copy() for _ in range(num_copies)]
    large_df = cudf.concat(df_list, ignore_index=True)
    large_df.index=cudf.Series(np.arange(len(large_df)))
    return large_df
    
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
        #df = copy_df(df, 2)
        loaded = datetime.datetime.now()
        print(f"Data loaded: {loaded - start}")
        sp = gpu_search_space(df, 'Class', 1, 5, spam_ignore)
        task = ps.gpu_task(sp, 1, depth=2, result_set_size=10)
        bfs = ps.gpu_bfs(task)
        sp_created = datetime.datetime.now()
        print(f'Search space created: {sp_created - loaded}')
        print(bfs.execute()['subgroup'])
        print(f"Execution time: {datetime.datetime.now() - sp_created}")
        print(f"Total time: {datetime.datetime.now() - start}")
        
    
