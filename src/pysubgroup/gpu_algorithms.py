from collections import deque
import cudf
import cupy as cp
import pandas
import pysubgroup as ps
from heapq import heappushpop, heappush

class gpu_task:
    def __init__(self, search_space, qf, result_set_size=10, depth=3, min_quality=float("-inf")):
        self.search_space = search_space
        self.qf = qf
        self.result_set_size = result_set_size
        self.depth = depth
        self.min_quality = min_quality

class gpu_sg:
    def __init__(self, length, selector_ids = [], quality=float("-inf"), optimistic_est = float("inf")):
        self.length = length
        self.selector_ids = selector_ids
        self.cover_arr = None
        self.quality = quality
        self.optimistic_est = optimistic_est
    
    def depth(self):
        return len(self.selector_ids)
    
    def get_tuple(self):
        return tuple(self.selector_ids)
    
    def __lt__(self, other):
        return self.quality < other.quality
    
class gpu_algorithm:
    
    def __init__(self, gpu_task):
        self.task = gpu_task
        self.selectors = self.task.search_space.sels.to_pandas()
        self.nr_sels = len(self.selectors)+1
        self.instances = self.task.search_space.instances
        self.quality = self.task.qf
        self.stats = self.task.search_space.stats
        self.result = []
    
    def eval_children(self, parent, sels, cache):
        if cache:
            old_arr = parent.cover_arr
        else:
            old_arr = self.stats.get_cover_arr_sels(parent.selector_ids)
        qualities, optimistics, arrs = self.stats.add_sels(old_arr, sels, self.quality)
        for i,q in enumerate(qualities):
            child = gpu_sg(self.instances, selector_ids=parent.selector_ids + [sels[i]])
            child.quality = q
            child.optimistic_est = optimistics[i]
            child.cover_arr = arrs[i]
            yield child
            
    def add_if_required(self, sg):
        if len(self.result) < self.task.result_set_size:
            heappush(self.result, (sg.quality, sg))
            return True
        else:
            if sg.optimistic_est > self.result[0][1].quality:
                heappushpop(self.result, (sg.quality, sg))
                return True
        return False
    
    def prepare_result(self, result):
        sgs = [sg for (_, sg) in result]
        sgs.sort(key=lambda x: x.quality, reverse=True)
        result1_df = pandas.DataFrame()
        result1_df['quality'] = [sg.quality for sg in sgs]
        result1_df['subgroup'] = [self.sels_to_string(sg.selector_ids) for sg in sgs]
        for sg in sgs:
            try:
                sg.cover_arr.shape
            except AttributeError:
                sg.cover_arr= self.stats.get_cover_arr_sels(sg.selector_ids)
        cover_arrs = cp.vstack([sg.cover_arr for sg in sgs])
        pos_arrs = (cover_arrs & self.task.search_space.reps[0])
        sizes = cp.sum(cover_arrs,axis = 1)
        hits = cp.sum(pos_arrs,axis = 1)
        stats = self.stats.compute_stats(sizes, hits)
        positives_dataset = cp.sum(self.task.search_space.reps[0])
        
        result2_df=cudf.DataFrame()
        result2_df['size_sg'] = stats['size_sg']
        result2_df['size_dataset'] = self.instances
        result2_df['positives_sg'] = stats['positives_sg']
        result2_df['positives_dataset'] = positives_dataset
        result2_df['size_complement'] = stats['size_complement']
        result2_df['relative_size_sg'] = stats['relative_size_sg']
        result2_df['relative_size_complement'] = stats['relative_size_complement']
        result2_df['coverage_sg'] = stats['coverage_sg']
        result2_df['coverage_complement'] = stats['coverage_complement']
        result2_df['target_share_sg'] = stats['target_share_sg']
        result2_df['target_share_complement'] = stats['target_share_complement']
        result2_df['target_share_dataset'] = positives_dataset/self.instances
        result2_df['lift'] = stats['lift']
        
        return pandas.concat([result1_df, result2_df.to_pandas()], axis=1)
    
    def sels_to_string(self, sels):
        strings = [self.sel_to_string(sel) for sel in sels]
        return ' AND '.join(strings)
    
    def sel_to_string(self, sel):
        selector = self.selectors.loc[sel]
        att = selector['attribute']
        low , high = selector['low'], selector['high']
        if selector['type'] == 'categorical':
            return f"{att} == {low}"
        else:
            return f"{att}: [{low}, {high}["
        
    
class gpu_dfs(gpu_algorithm):
    def __init__(self, gpu_task):
        super().__init__(gpu_task)
        
    def execute(self, root = [], caching = False, pruning = True):
        root = gpu_sg(self.instances, selector_ids=root)
        if caching:
            root.cover_arr = cp.ones(root.length, dtype=bool)
        stack = deque([root])
        explored_nodes={root.get_tuple: root.quality}
        
        while stack:
            current = stack.pop()
            print(current.selector_ids)
            
            if current.depth() < self.task.depth:
                new_sels = get_child_sels(current, self.selectors)
                for child in self.eval_children(current, new_sels, caching):
                    if child.get_tuple() not in explored_nodes:
                        explored_nodes[child.get_tuple()] = child.quality
                        added = self.add_if_required(child)
                        if (added or (not pruning)) and current.depth() < self.task.depth - 1:
                            stack.append(child)
        return self.prepare_result(self.result)
    
class apriori_gpu(gpu_algorithm):
    def __init__(self, gpu_task):
        super().__init__(gpu_task)

    
                
#TODO: delete, just needs way more memory than dfs
class gpu_bfs:
    def __init__(self, gpu_task):
        self.task = gpu_task
        self.selectors = self.task.search_space.sels.to_pandas()
        self.nr_sels = len(self.selectors)+1
        
    def execute(self, sel_ids = (), batch_size=1024):
        root = gpu_sg(sel_ids)
        queue = deque([root])
        eval_buffer = []
        explored_nodes = {root.selector_ids: root.quality}
        
        while queue:
            current = queue.popleft()
            eval_buffer.append(current)
            
            
            if len(eval_buffer) >= batch_size:
                print(len(eval_buffer), len(queue))
                sgs = eval_gpu(eval_buffer, self.task.search_space, self.task.qf)
                process_results(sgs, explored_nodes)
                eval_buffer.clear()
            
            if current.depth() < self.task.depth:
                children = get_children(current, self.selectors)
                for child in children:
                    if child.selector_ids not in explored_nodes:
                        queue.append(child)
                        explored_nodes[child.selector_ids]=child.quality
        
        if eval_buffer:
            sgs = eval_gpu(eval_buffer, self.task.search_space, self.task.qf)
            process_results(sgs, explored_nodes)
            eval_buffer.clear()
            
        return explored_nodes
        

def get_children(parent, sels):
    parent_sg = parent.selector_ids
    parent_size = parent.length
    sels_to_add = get_child_sels(parent, sels)
    return list(gpu_sg(parent_size, parent_sg + [new_sel]) for new_sel in sels_to_add)

def get_child_sels(parent, sels):
    parent_sg = parent.selector_ids
    attributes = sels.loc[parent_sg]['attribute'].unique().tolist()
    sels_to_add = sels[~sels['attribute'].isin(attributes)]['id'].tolist()
    if 0 in sels_to_add:
        sels_to_add.remove(0)
    return sels_to_add
    
#TODO: delete all these
def tuples_to_ints(sgs, nr_sels):
    row_indices = cp.repeat(cp.arange(len(sgs)), [t.depth() for t in sgs])
    col_indices = cp.array([item for sg in sgs for item in sg.selector_ids], dtype=int)
    result = cp.zeros((len(sgs), nr_sels), dtype=bool)
    result[row_indices, col_indices] = True
    return ps.bool_to_int(result)

def eval_gpu(sgs, search_space, qf):
    sg_ints = tuples_to_ints(sgs, search_space.sels.shape[0])
    stats = search_space.stats.compute_stats_subgroups(sg_ints)
    qualities = search_space.stats.compute_quality(stats, qf).to_arrow().to_pylist()
    for i, quality in enumerate(qualities):
        sgs[i].quality = quality
    return sgs

def process_results(sgs, explored_nodes):
    for sg in sgs:
        explored_nodes[sg.selector_ids] = sg.quality