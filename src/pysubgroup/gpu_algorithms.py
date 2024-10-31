import cudf
import cupy as cp
import pandas
from typing import Tuple


class GpuTask:
    # Store all relevant parameters for a SG discovery task
    def __init__(
        self, search_space, qf, result_set_size=10, depth=3, min_quality=float("-inf")
    ):
        self.search_space = search_space
        self.qf = qf
        self.result_set_size = result_set_size
        self.depth = depth
        self.min_quality = min_quality


class VisitedVector:
    # Vectorized hash map for storing already visited subgroups
    # Disclaimer: Claude wrote this class
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
                    table[n, r] = table[n - 1, r - 1] + table[n - 1, r]
        return table

    def _calculate_width(self):
        return sum(self._combinations(self.sels, i) for i in range(1, self.depth + 1))

    def _combinations(self, n, r):
        if r > n:
            return 0
        return int(cp.prod(cp.arange(n, n - r, -1)) // cp.prod(cp.arange(1, r + 1)))

    # nodes should already be sorted here
    def _hash(self, node):
        index = cp.uint32(0)
        for i, val in enumerate(node):
            smaller_combinations = cp.sum(self.comb_table[self.sels, : i + 1])
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
            smaller_combinations = cp.sum(self.comb_table[self.sels, : i + 1])
            indices[mask] += (
                smaller_combinations + self.comb_table[nodes[mask, i] - 1, i + 1]
            )
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


class ChunkedVisitedVector:
    # Sparse/Chunked version of the above
    # Disclaimer: Claude also wrote this class
    def __init__(self, sels: int, depth: int, chunk_size: int = 1024):
        self.sels = sels
        self.depth = depth
        self.chunk_size = chunk_size
        self.comb_table = self._precompute_combinations()
        self.chunks = {}  # Dictionary of chunk_id -> bit array

    def _precompute_combinations(self) -> cp.ndarray:
        """Precompute combination table for faster hashing"""
        table = cp.zeros((self.sels + 1, self.depth + 1), dtype=cp.uint32)
        for n in range(self.sels + 1):
            for r in range(self.depth + 1):
                if r == 0 or n == 0:
                    table[n, r] = 1
                else:
                    table[n, r] = table[n - 1, r - 1] + table[n - 1, r]
        return table

    def _hash_node(self, node: cp.ndarray) -> cp.uint32:
        """Hash a single node (sorted array of integers)"""
        index = cp.uint32(0)
        valid_elements = node[node != 0]
        for i, val in enumerate(valid_elements):
            smaller_combinations = cp.sum(self.comb_table[self.sels, : i + 1])
            index += smaller_combinations + self.comb_table[val - 1, i + 1]
        return index

    def _vectorized_hash(self, nodes: cp.ndarray) -> cp.ndarray:
        """Hash multiple nodes in parallel"""
        if nodes.ndim == 1:
            nodes = nodes.reshape(1, -1)

        # Sort each row
        nodes = cp.sort(nodes, axis=1)

        # Calculate indices
        indices = cp.zeros(len(nodes), dtype=cp.uint32)
        for i in range(nodes.shape[1]):
            mask = nodes[:, i] != 0
            if cp.any(mask):
                smaller_combinations = cp.sum(self.comb_table[self.sels, : i + 1])
                indices[mask] += (
                    smaller_combinations + self.comb_table[nodes[mask, i] - 1, i + 1]
                )
        return indices

    def _get_chunk_id_and_offset(
        self, hashed_indices: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Convert hashed indices to chunk IDs and offsets"""
        chunk_ids = hashed_indices // self.chunk_size
        offsets = hashed_indices % self.chunk_size
        return chunk_ids, offsets

    def _ensure_chunks_exist(self, chunk_ids: cp.ndarray) -> None:
        """Create chunks if they don't exist"""
        unique_chunks = cp.unique(chunk_ids).get()
        for chunk_id in unique_chunks:
            if chunk_id not in self.chunks:
                array_size = (self.chunk_size - 1) // 32 + 1
                self.chunks[chunk_id] = cp.zeros(array_size, dtype=cp.uint32)

    def set_visited(self, nodes: cp.ndarray) -> None:
        """Mark nodes as visited"""
        hashed_indices = self._vectorized_hash(nodes)
        self.set_visited_hashed(hashed_indices)

    def set_visited_hashed(self, hashed_indices: cp.ndarray) -> None:
        """Mark pre-hashed indices as visited"""
        chunk_ids, offsets = self._get_chunk_id_and_offset(hashed_indices)
        self._ensure_chunks_exist(chunk_ids)

        # Process each chunk
        unique_chunks = cp.asnumpy(cp.unique(chunk_ids))
        for chunk_id in unique_chunks:
            chunk_mask = chunk_ids == chunk_id
            chunk_offsets = offsets[chunk_mask]
            cp.bitwise_or.at(
                self.chunks[chunk_id], chunk_offsets // 32, 1 << (chunk_offsets % 32)
            )

    def is_visited(self, nodes: cp.ndarray) -> cp.ndarray:
        """Check if nodes have been visited"""
        hashed_indices = self._vectorized_hash(nodes)
        return self.is_visited_hashed(hashed_indices)

    def is_visited_hashed(self, hashed_indices: cp.ndarray) -> cp.ndarray:
        """Check if pre-hashed indices have been visited"""
        chunk_ids, offsets = self._get_chunk_id_and_offset(hashed_indices)
        result = cp.zeros(len(hashed_indices), dtype=cp.bool_)

        for chunk_id in cp.asnumpy(cp.unique(chunk_ids)):
            if chunk_id in self.chunks:
                chunk_mask = chunk_ids == chunk_id
                chunk_offsets = offsets[chunk_mask]
                result[chunk_mask] = (
                    self.chunks[chunk_id][chunk_offsets // 32]
                    & (1 << (chunk_offsets % 32))
                ) != 0

        return result

    def memory_usage(self) -> int:
        """Return approximate memory usage in bytes"""
        num_chunks = len(self.chunks)
        chunk_memory = sum(chunk.nbytes for chunk in self.chunks.values())
        comb_table_memory = self.comb_table.nbytes
        return chunk_memory + comb_table_memory


class GpuAlgorithm:

    def __init__(self, GpuTask, apriori, sparse):

        self.task = GpuTask
        self.selectors = self.task.search_space.sels
        self.nr_sels = len(self.selectors)
        self.instances = self.task.search_space.instances
        self.quality = self.task.qf
        self.stats = self.task.search_space.stats
        self.apriori = apriori

        self.result = cudf.DataFrame({"quality": 0})
        for i in range(self.task.depth):
            col_name = f"sel_{i}"
            self.result[col_name] = 0

        # Apriori needs a 2nd visited vector to keep track of pruning
        if sparse:
            self.visited = ChunkedVisitedVector(self.nr_sels, self.task.depth)
            if apriori:
                self.apriori_vec = ChunkedVisitedVector(self.nr_sels, self.task.depth)
        else:
            self.visited = VisitedVector(self.nr_sels, self.task.depth)
            if apriori:
                self.apriori_vec = VisitedVector(self.nr_sels, self.task.depth)

    def add_if_required(self, sgs, qualities, optimistics):
        # Check if newly evaluated subgroups are better than previous best
        min_q = self.result["quality"].min()
        addeds = qualities > min_q
        to_add = cudf.DataFrame({"quality": qualities[addeds]})
        lst = []
        for i in range(self.task.depth):
            col_name = f"sel_{i}"
            to_add[col_name] = sgs[addeds, i]
            lst.append(col_name)
        self.result = cudf.concat([self.result, to_add])
        # Sort result set and only keep n best
        self.result = self.result.sort_values(by="quality", ascending=False).nlargest(
            self.task.result_set_size, columns="quality"
        )

        # Optimistic estimate pruning
        to_explore = optimistics > min_q
        return to_explore

    # Below is only for formatting results
    def prepare_result(self, result):
        result1_df = pandas.DataFrame()
        sgs = [
            [item for item in sg if item != 0]
            for sg in result.drop(columns="quality").to_pandas().values.tolist()
        ]
        result1_df["quality"] = result["quality"].to_pandas()
        result1_df["subgroup"] = [self.sels_to_string(sg) for sg in sgs]
        result1_df.reset_index(inplace=True, drop=True)
        cover_arrs = cp.vstack([self.stats.get_cover_arr_sels(sg) for sg in sgs])
        pos_arrs = cover_arrs & self.task.search_space.reps[0]
        sizes = cp.sum(cover_arrs, axis=1)
        hits = cp.sum(pos_arrs, axis=1)
        stats = self.stats.compute_stats(sizes, hits)
        positives_dataset = cp.sum(self.task.search_space.reps[0])
        result2_df = cudf.DataFrame()
        result2_df["size_sg"] = stats["size_sg"]
        result2_df["size_dataset"] = self.instances
        result2_df["positives_sg"] = stats["positives_sg"]
        result2_df["positives_dataset"] = positives_dataset
        result2_df["size_complement"] = stats["size_complement"]
        result2_df["relative_size_sg"] = stats["relative_size_sg"]
        result2_df["relative_size_complement"] = stats["relative_size_complement"]
        result2_df["coverage_sg"] = stats["coverage_sg"]
        result2_df["coverage_complement"] = stats["coverage_complement"]
        result2_df["target_share_sg"] = stats["target_share_sg"]
        result2_df["target_share_complement"] = stats["target_share_complement"]
        result2_df["target_share_dataset"] = positives_dataset / self.instances
        result2_df["lift"] = stats["lift"]
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
        return " AND ".join(strings)

    def sel_to_string(self, sel):
        selector = self.selectors.loc[sel]
        att = selector["attribute"].iloc[0]
        low, high = selector["low"].iloc[0], selector["high"].iloc[0]
        if selector["type"].eq("categorical").any():
            return att + "==" + str(low)
        else:
            return att + ":[" + str(low) + "," + str(high) + "["


class GpuDfs(GpuAlgorithm):
    def __init__(self, GpuTask, apriori=False, sparse=True):
        super().__init__(GpuTask, apriori, sparse)

    def eval_children(self, parent, sels, depth):
        # Calculate stats for all relevant children of a subgroup
        # First, stack parent and replace 0s with the new selectors
        children = cp.vstack([parent] * sels.size)
        children[:, depth] = sels

        # Check if any were already visited before
        visited = self.visited.is_visited(children)
        children = children[~visited]

        # Maybe no children left?
        if children.shape[0] == 0:
            return children, None, None
        self.visited.set_visited(children)

        qualities, optimistics = self.stats.compute_quality_optimistic(
            children, self.quality
        )
        return children, qualities, optimistics

    def get_child_sels(self, parent):
        # Get all possible child selectors for a parent by filtering all selectors that share an attribute
        sels = self.selectors
        attributes = sels.loc[parent]["attribute"].unique()
        sels_to_add = cp.from_dlpack(
            sels[~sels["attribute"].isin(attributes)]["id"].to_dlpack()
        )
        return sels_to_add[sels_to_add != 0].astype(cp.uint32)

    def apriori_pruning(self, sg, sels):
        # Apriori pruning for DFS is essentially a big OR, finding any child that has a pruned parent
        # Unlike BFS, can't assume every parent was visited before
        remove = cp.zeros(sels.shape[0], dtype=bool)
        depth = sg[sg != 0].shape[0]

        for d in range(depth):
            to_check = cp.copy(sg)
            to_check = cp.tile(to_check, (sels.shape[0], 1))
            # Iteratively replace one selector in parent with the vector of new selectors, and check in apriori vector
            to_check[:, d] = sels
            # Here is the OR
            remove = remove | self.apriori_vec.is_visited(to_check)

        return sels[~remove]

    def execute(self):

        # Empty subgroup is root, set it as visited
        stack = cp.array([[0] * self.task.depth], dtype=cp.uint16)
        self.visited.set_visited(cp.array([[0] * self.task.depth]))

        while stack.size > 0:

            current = stack[-1]
            stack = stack[:-1]

            depth = current[current != 0].shape[0]
            new_sels = self.get_child_sels(current)

            if self.apriori:
                new_sels = self.apriori_pruning(current, new_sels)

            # If all children could get pruned, skip this node
            if new_sels.size == 0:
                continue

            children, qualities, optimistics = self.eval_children(
                current, new_sels, depth
            )
            # If all children visited before, skip
            if children.shape[0] == 0:
                continue

            # Optimistic estimate pruning
            addeds = self.add_if_required(children, qualities, optimistics)

            # If max depth not reached, add good children to stack
            if depth < self.task.depth - 1:
                stack = cp.append(stack, children[addeds], axis=0)
                if self.apriori:
                    self.apriori_vec.set_visited(children[~addeds])

        return self.prepare_result(self.result)


class GpuBfs(GpuAlgorithm):

    def __init__(self, GpuTask, apriori=False, sparse=True):

        super().__init__(GpuTask, apriori, sparse)
        # This stores attributes from original data as ints for easier lookup later
        self.int_attributes = self.sels_attributes_ints()

    def sels_attributes_ints(self):
        # Return map holding attribute-int pairs
        attributes = self.selectors["attribute"].unique()
        att_int = {att: i for i, att in enumerate(attributes.to_pandas())}
        return cp.array(self.selectors["attribute"].map(att_int))

    def get_next_level(self, parents, depth):
        # add ALL selectors to previous by copying parents multiple times
        sels = self.int_attributes
        nr_sels = sels.shape[0]

        added_sels = cp.tile(cp.arange(1, nr_sels, dtype=cp.uint16), parents.shape[0])
        children = cp.repeat(parents, nr_sels - 1, axis=0)
        children[:, depth - 1] = added_sels

        # Eliminate subgroups generated this way that have two selectors for the same attribute
        mask = cp.ones((children.shape[0]), dtype=bool)
        added_att = sels[children[:, depth - 1]]

        for i in range(depth - 1):
            attributes_i = sels[children[:, i]]
            mask &= attributes_i != added_att
        children = children[mask]

        return children

    def apriori_pruning(self, sgs, depth):
        # Check all parents of subgroups and eliminate those that have a pruned one
        keep = cp.ones((sgs.shape[0]), dtype=bool)

        for d in range(depth):
            to_check = cp.copy(sgs)
            to_check[:, d] = 0
            keep &= self.apriori_vec.is_visited(to_check)

        return sgs[keep]

    def remove_dupes(self, sgs):
        # Remove duplicate subgroups from a set
        vec = self.visited

        # First, make sure we haven't seen any of the subgroups before
        sgs = sgs[~vec.is_visited(sgs)]

        # Hash all subgroups and add them to vector
        hashes = vec._vectorized_hash(sgs)
        vec.set_visited_hashed(hashes)

        # Find uniques and drop all others
        _, ids = cp.unique(hashes, return_index=True)
        return sgs[ids]

    def chunk_size_parents(self, safety=0.8):
        # Roughly estimated chunk size for child generation
        mem = cp.cuda.runtime.memGetInfo()[0]
        return int((mem * safety) / (self.nr_sels * 8))

    def chunk_size_children(self, safety=0.8):
        # Roughly estimated chunk size for subgroup evaluation
        mem = cp.cuda.runtime.memGetInfo()[0]
        sg_size = self.instances * 3
        return int((mem * safety) / (sg_size))

    def execute(self):

        parent_cs = self.chunk_size_parents()
        child_cs = self.chunk_size_children()

        # Level 0 is empty
        sg_levels = [[]]

        # Level 1 is just the selectors themselves
        sg_1 = cp.zeros((self.nr_sels - 1, self.task.depth), dtype=cp.uint16)
        sg_1[:, 0] = cp.arange(1, self.nr_sels)
        self.visited.set_visited(sg_1)

        # Their stats were computed already
        stats_1 = self.stats.compute_stats_sels().drop([0])
        q_1 = self.stats.compute_quality(stats_1, self.quality)
        o_1 = self.stats.compute_optimistic(stats_1, self.quality)
        self.add_if_required(sg_1, q_1, o_1)

        # Unlikely but just in case a selector is bad enough for optimistic estimate pruning
        min_q = self.result["quality"].min()
        to_add = sg_1[o_1 > min_q]
        if self.apriori:
            self.apriori_vec.set_visited(to_add)
        sg_levels.append(to_add)

        # Iterate through levels
        for depth in range(2, self.task.depth + 1):
            parents = sg_levels[depth - 1]
            sgs_at_depth = []

            # chunk making children, this is memory intensive and apriori might eliminate enough sgs in the process to keep discovery going
            children = []

            for i in range(0, parents.shape[0], parent_cs):

                chunk = parents[i : i + parent_cs]
                new_children = self.get_next_level(chunk, depth)

                if self.apriori:
                    new_children = self.apriori_pruning(new_children, depth)

                children.append(new_children)

            # Apriori might get rid of all children, at which point exploration is done
            if not children:
                return self.prepare_result(self.result)

            # Put children in one array
            children = cp.concatenate(children, axis=0)

            # Iterate through chunks for sg evaluation
            for j in range(0, children.shape[0], child_cs):

                chunk = children[j : j + child_cs]
                chunk = self.remove_dupes(chunk)
                # If all children in a chunk were already explored, skip it
                if chunk.size == 0:
                    continue

                qualities, optimistics = self.stats.compute_quality_optimistic(
                    chunk, self.quality
                )

                # Optimistic estimate pruning
                to_explore = self.add_if_required(chunk, qualities, optimistics)

                # Set all the added subgroups as not pruned in Apriori vector
                if self.apriori:
                    self.apriori_vec.set_visited(chunk[to_explore])

                sgs_at_depth.append(chunk[to_explore])

            # If an entire level was pruned, stop
            if not sgs_at_depth:
                return self.prepare_result(self.result)

            sg_levels.append(cp.concatenate(sgs_at_depth, axis=0))

        return self.prepare_result(self.result)
