import cudf
import cupy as cp
import pandas

class gpu_task:
    def __init__(self, search_space, qf, result_set_size=10, depth=3, min_quality=float("-inf")):
        self.search_space = search_space
        self.qf = qf
        self.result_set_size = result_set_size
        self.depth = depth
        self.min_quality = min_quality

class visited_vector:
    def __init__(self, sels, depth):
        self.sels = sels
        self.depth = depth
        self.width = self._calculate_width()
        self.array_size = (self.width - 1) // 32 + 1
        self.bit_array = cp.zeros(self.array_size, dtype=cp.uint32)
        self.comb_table = self._precompute_combinations()
        
    def _precompute_combinations(self):
        table = cp.zeros((self.sels + 1, self.depth + 1), dtype=cp.uint32)
        for n in range(self.sels + 1):
            for r in range(self.depth + 1):
                if r == 0 or n == 0:
                    table[n, r] = 1
                else:
                    table[n, r] = table[n-1, r-1] + table[n-1, r]
        return table
            
    def _calculate_width(self):
        return sum(self._combinations(self.sels, i) for i in range(1, self.depth + 1))
        
    def _combinations(self, n, r):
        if r > n:
            return 0
        return int(cp.prod(cp.arange(n, n-r, -1)) // cp.prod(cp.arange(1, r+1)))
    
    #nodes should already be sorted here
    def _hash(self, node):
        index = cp.uint32(0)
        for i, val in enumerate(node):
            smaller_combinations = cp.sum(self.comb_table[self.sels, :i+1])
            index += smaller_combinations + self.comb_table[val - 1, i + 1]
        return index
    
    def _vectorized_hash(self, nodes):
        # Ensure nodes is 2D
        if nodes.ndim == 1:
            nodes = nodes.reshape(1, -1)
        # Sort each row
        nodes = cp.sort(nodes, axis=1)
        
        # Calculate indices
        indices = cp.zeros(len(nodes), dtype=cp.uint32)
        for i in range(nodes.shape[1]):
            mask = nodes[:, i] != 0
            smaller_combinations = cp.sum(self.comb_table[self.sels, :i+1])
            indices[mask] += smaller_combinations + self.comb_table[nodes[mask, i] - 1, i + 1]
        return indices
    
    def set_visited(self, nodes):
        indices = self._vectorized_hash(nodes)
        cp.bitwise_or.at(self.bit_array, indices // 32, 1 << (indices % 32))
    
    def set_visited_hashed(self, nodes_hashed):
        cp.bitwise_or.at(self.bit_array, nodes_hashed // 32, 1 << (nodes_hashed % 32))

    def is_visited(self, nodes):
        indices = self._vectorized_hash(nodes)
        return (self.bit_array[indices // 32] & (1 << (indices % 32))) != 0
    
    def is_visited_hashed(self, nodes_hashed):
        return (self.bit_array[nodes_hashed // 32] & (1 << (nodes_hashed % 32))) != 0
    
class gpu_algorithm:
    
    def __init__(self, gpu_task, apriori):
        self.task = gpu_task
        self.selectors = self.task.search_space.sels
        self.nr_sels = len(self.selectors)
        self.instances = self.task.search_space.instances
        self.quality = self.task.qf
        self.stats = self.task.search_space.stats
        self.result = cudf.DataFrame({'quality': 0})
        for i in range(self.task.depth):
            col_name = f'sel_{i}'
            self.result[col_name] = 0
        self.visited = visited_vector(self.nr_sels, self.task.depth)
        self.apriori = apriori
        if apriori:
            self.apriori_vec = visited_vector(self.nr_sels, self.task.depth)
        else:
            self.apriori_vec = None
        
    
    def eval_children(self, parent, sels, depth):
        qualities, optimistics = self.stats.add_sels(parent, sels, self.quality)
        children = cp.vstack([parent]*sels.size)
        children[:,depth] = sels
        return children, qualities, optimistics
            
    def add_if_required(self, sgs, qualities, optimistics):
        min_q = self.result['quality'].min()
        to_explore = optimistics > min_q
        addeds = qualities > min_q
        
        to_add = cudf.DataFrame({'quality' : qualities[addeds]})
        lst = []
        for i in range(self.task.depth):
            col_name = f'sel_{i}'
            to_add[col_name] = sgs[addeds,i]
            lst.append(col_name)
            
        
        self.result = cudf.concat([self.result, to_add])
        self.result = self.result.sort_values(by='quality', ascending=False).nlargest(self.task.result_set_size, columns='quality')
        return to_explore
    
    def prepare_result(self, result):
        result1_df = pandas.DataFrame()
        sgs = [[item for item in sg if item != 0] for sg in result.drop(columns = 'quality').to_pandas().values.tolist()]
        result1_df['quality'] = result['quality'].to_pandas()
        result1_df['subgroup'] = [self.sels_to_string(sg) for sg in sgs]
        result1_df.reset_index(inplace=True, drop=True)
        cover_arrs = cp.vstack([self.stats.get_cover_arr_sels(sg) for sg in sgs])
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
        df = pandas.concat([result1_df, result2_df.to_pandas()], axis=1)
        return df.round(
        {
            "quality": 3,
            "size_sg": 0,
            "size_dataset": 0,
            "positives_sg": 0,
            "positives_dataset": 0,
            "size_complement": 0,
            "relative_size_sg": 3,
            "relative_size_complement": 3,
            "coverage_sg": 3,
            "coverage_complement": 3,
            "target_share_sg": 3,
            "target_share_complement": 3,
            "target_share_dataset": 3,
            "lift": 3,
        }
    )
    
    def sels_to_string(self, sels):
        strings = [self.sel_to_string(sel) for sel in sels if sel != 0]
        return ' AND '.join(strings)
    
    def sel_to_string(self, sel):
        selector = self.selectors.loc[sel]
        att = selector['attribute'].iloc[0]
        low , high = round(selector['low'].iloc[0],3), round(selector['high'].iloc[0],3)
        if selector['type'].eq('categorical').any():
            return att + '==' + str(low)
        else:
            return att + ':[' + str(low) + ',' + str(high) + '['
        
    
class gpu_dfs(gpu_algorithm):
    def __init__(self, gpu_task, apriori=False):
        super().__init__(gpu_task, apriori)
        
    def execute(self):
        stack =cp.array([[0]*self.task.depth] , dtype=cp.uint16)
        self.visited.set_visited(cp.array([[0]*self.task.depth]))
        
        while stack.size > 0:
            
            current = stack[-1]
            stack = stack[:-1]
            depth = cp.sum(current != 0)
            new_sels = get_child_sels(current, self.selectors)
            if self.apriori:
                new_sels = apriori_pruning_dfs(current, new_sels, self.apriori_vec)
            #If all children could get pruned, skip this node
            if new_sels.size == 0:
                continue
            children, qualities, optimistics = self.eval_children(current, new_sels, depth)
            vis = self.visited.is_visited(children)
            children=children[~vis]                
            qualities=qualities[~vis]
            optimistics=optimistics[~vis]
            self.visited.set_visited(children)
            addeds = self.add_if_required(children, qualities, optimistics)
            
            if depth < self.task.depth - 1:
                stack = cp.append(stack, children[addeds], axis = 0)
                if self.apriori:
                    self.apriori_vec.set_visited(children[~addeds])
                    
        return self.prepare_result(self.result)
    
class gpu_bfs(gpu_algorithm):
    def __init__(self, gpu_task, apriori = False):
        super().__init__(gpu_task, apriori)
        self.int_attributes = self.sels_attributes_ints()
        
    def sels_attributes_ints(self):
        attributes = self.selectors['attribute'].unique()
        att_int = {att:i for i,att in enumerate(attributes.to_pandas())}
        return cp.array(self.selectors['attribute'].map(att_int))

    def execute(self):
        #Level 0 is empty
        sg_levels = [[]]
        
        #level 1 is just the selectors themselves:
        sg_1  = cp.zeros((self.nr_sels-1,self.task.depth), dtype=cp.uint16)
        sg_1[:,0]=cp.arange(1,self.nr_sels)
        self.visited.set_visited(sg_1)
        #calculated stats already
        stats_1 = self.stats.compute_stats_sels().drop([0])
        q_1 = self.stats.compute_quality(stats_1, self.quality)
        o_1 = self.stats.compute_optimistic(stats_1, self.quality)
        self.add_if_required(sg_1, q_1, o_1)
        #unlikely but just in case:
        min_q = self.result['quality'].min()
        to_add = sg_1[o_1 > min_q]
        if self.apriori:
            self.apriori_vec.set_visited(to_add)
        sg_levels.append(to_add)
        
        #iterate through levels
        for depth in range(2, self.task.depth+1):
            parents = sg_levels[depth-1]
            parent_cs = chunk_size_parents(self.nr_sels)
            child_cs = chunk_size_children(self.instances)
            sg_depth = []
            
            #chunk making children, this is very memory intensive and apriori might eliminate enough sgs in the process to keep discovery going
            children=[]
            for i in range(0, parents.shape[0], parent_cs):
                chunk = parents[i:i+parent_cs]
                new_children = get_next_level(chunk, self.int_attributes, depth, self.visited)
                if self.apriori:
                    new_children = apriori_pruning_bfs(new_children, self.apriori_vec, depth)
                
                children.append(new_children)
            #apriori might get rid of all children
            if not children:
                return self.prepare_result(self.result)
            children=cp.concatenate(children, axis=0)
                  
            #iterate through chunks for sg evaluation
            for j in range(0, children.shape[0], child_cs):
                chunk = children[j:j+child_cs]
                chunk = remove_dupes(chunk, self.visited)
                if chunk.size == 0:
                    continue
                qualities, optimistics = self.stats.compute_quality_optimistic(chunk,self.quality)
                to_explore = self.add_if_required(chunk, qualities, optimistics)
                if self.apriori:
                    self.apriori_vec.set_visited(chunk[to_explore])
                sg_depth.append(chunk[to_explore])
                    
            sg_levels.append(cp.concatenate(sg_depth, axis=0))
                
                
        return self.prepare_result(self.result)
                       
    
def get_child_sels(parent, sels):
    attributes = sels.loc[parent]['attribute'].unique()
    sels_to_add = cp.from_dlpack(sels[~sels['attribute'].isin(attributes)]['id'].to_dlpack())
    return sels_to_add[sels_to_add != 0].astype(cp.uint32)

def get_next_level(parents, sels, depth, vec):
    #add ALL sels to previous:
    nr_sels = sels.shape[0]
    added_sels = cp.tile(cp.arange(1, nr_sels, dtype=cp.uint16), parents.shape[0])
    children = cp.repeat(parents, nr_sels-1, axis=0)
    children[:,depth-1] = added_sels
    #eliminate duplicate attributes
    mask = cp.ones((children.shape[0]), dtype=bool)
    added_att = sels[children[:,depth-1]]
    for i in range(depth-1):
        attributes_i = sels[children[:,i]]
        mask &= attributes_i != added_att
    children = children[mask]  
    return children

def chunk_size_parents(nr_sels, safety = .8):
    mem = cp.cuda.runtime.memGetInfo()[0]
    return int((mem*safety)/(nr_sels*8))

def chunk_size_children(nr_instances, safety = .8):
    mem = cp.cuda.runtime.memGetInfo()[0]
    sg_size = nr_instances*3
    return int((mem*safety)/(sg_size))

def remove_dupes(children, vec):
    children = children[~vec.is_visited(children)]
    hashes = vec._vectorized_hash(children)
    vec.set_visited_hashed(hashes)
    _, ids = cp.unique(hashes, return_index=True)
    return children[ids]
    
#for bfs: check if all depth-1 subsets are frequent
def apriori_pruning_bfs(sgs, apriori_vec, depth):
    keep = cp.ones((sgs.shape[0]), dtype=bool)
    for d in range(depth):
        to_check = cp.copy(sgs)
        to_check[:,d] = 0
        keep &= apriori_vec.is_visited(to_check)
    return sgs[keep]

#for dfs: check if no depth-1 subset is infrequent
def apriori_pruning_dfs(sg, sels, apriori_vec):
    remove = cp.zeros(sels.shape[0], dtype=bool)
    depth = sg[sg!=0].shape[0]
    for d in range(depth):
        to_check = cp.copy(sg)
        to_check = cp.tile(to_check, (sels.shape[0],1))
        to_check[:,d] = sels
        remove = remove | apriori_vec.is_visited(to_check)
        
    return sels[~remove]