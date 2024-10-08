import numpy as np
import cupy as cp
from numba import cuda
import cudf
import pandas as pd


from pysubgroup.subgroup_description import Conjunction, Disjunction


class RepresentationBase:
    def __init__(self, new_conjunction, selectors_to_patch):
        self._new_conjunction = new_conjunction
        self.previous_conjunction = None
        self.selectors_to_patch = selectors_to_patch

    def patch_all_selectors(self):
        for sel in self.selectors_to_patch:
            self.patch_selector(sel)

    def patch_selector(self, sel):  # pragma: no cover
        raise NotImplementedError()  # pragma: no cover

    def patch_classes(self):
        pass

    def undo_patch_classes(self):
        pass

    def __enter__(self):
        self.patch_classes()
        self.patch_all_selectors()
        return self

    def __exit__(self, *args):
        self.undo_patch_classes()


class BitSet_Conjunction(Conjunction):
    n_instances = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation = self.compute_representation()

    def compute_representation(self):
        # empty description ==> return a list of all '1's
        if not self._selectors:
            return np.full(BitSet_Conjunction.n_instances, True, dtype=bool)
        # non-empty description
        return np.all([sel.representation for sel in self._selectors], axis=0)

    @property
    def size_sg(self):
        return np.count_nonzero(self.representation)

    def append_and(self, to_append):
        super().append_and(to_append)
        self.representation = np.logical_and(
            self.representation, to_append.representation
        )

    @property
    def __array_interface__(self):
        return self.representation.__array_interface__


class BitSet_Disjunction(Disjunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation = self.compute_representation()

    def compute_representation(self):
        # empty description ==> return a list of all '1's
        if not self._selectors:
            return np.full(BitSet_Conjunction.n_instances, False, dtype=bool)
        # non-empty description
        return np.any([sel.representation for sel in self._selectors], axis=0)

    @property
    def size_sg(self):
        return np.count_nonzero(self.representation)

    def append_or(self, to_append):
        super().append_or(to_append)
        self.representation = np.logical_or(
            self.representation, to_append.representation
        )

    @property
    def __array_interface__(self):
        return self.representation.__array_interface__


class BitSetRepresentation(RepresentationBase):
    Conjunction = BitSet_Conjunction
    Disjunction = BitSet_Disjunction

    def __init__(self, df, selectors_to_patch):
        self.df = df
        super().__init__(BitSet_Conjunction, selectors_to_patch)

    def patch_selector(self, sel):
        sel.representation = sel.covers(self.df)
        sel.size_sg = np.count_nonzero(sel.representation)

    def patch_classes(self):
        BitSet_Conjunction.n_instances = len(self.df)
        super().patch_classes()
        
class CUDABitSet_Conj(Conjunction):
    n_instances = 0
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation = self.compute_representation()
        
    def compute_representation(self):
        if not self.selectors:
            return cp.full(self.n_instances, True, dtype=bool)
        return cp.all(cp.stack([sel.representation for sel in self.selectors]), axis=0)
    
    def append_and(self, to_append):
        super().append_and(to_append)
        self.representation = cp.logical_and(self.representation, to_append.representation)
        
    @property
    def size_sg(self):
        return cp.count_nonzero(self.representation)
    
    @property
    def __array_interface__(self):
        return self.representation.__cuda_array_interface__
    
    @property
    def __cuda_array_interface__(self):
        return self.representation.__cuda_array_interface__
        
    
class CUDABitSet_Disj(Disjunction):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation = self.compute_representation()
        
    def compute_representation(self):
        if not self.selectors:
            return cp.full(CUDABitSet_Conj.n_instances, False, dtype=bool)
        return cp.any([sel.representation for sel in self.selectors],axis=0)
    
    def append_or(self, to_append):
        super().append_or(to_append)
        self.representation=cp.logical_or(self.representation, to_append.representation)
        
    @property
    def size_sg(self):
        return cp.count_nonzero(self.representation)
    
    @property
    def __array_interface__(self):
        return self.representation.__cuda_array_interface__
    
    @property
    def __cuda_array_interface__(self):
        return self.representation.__cuda_array_interface__
        
    
class CUDABitSetRepr(RepresentationBase):
    Conjunction = CUDABitSet_Conj
    Disjunction = CUDABitSet_Disj
    
    def __init__(self, df, selectors_to_patch):
        if not isinstance(df, cudf.DataFrame): 
            raise TypeError("use a cudf dataframe for gpu acceleration")
        self.df=df
        super().__init__(CUDABitSet_Conj, selectors_to_patch)
    
    def patch_selector(self, sel):
        sel.representation = sel.covers(self.df)
        sel.size_sg = cp.count_nonzero(sel.representation)
    
    def patch_classes(self):
        CUDABitSet_Conj.n_instances=len(self.df)
        super().patch_classes()


class Set_Conjunction(Conjunction):
    all_set = set()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation = self.compute_representation()
        self.arr_for_interface = np.array(list(self.representation), dtype=int)

    def compute_representation(self):
        # empty description ==> return a list of all '1's
        if not self._selectors:
            return Set_Conjunction.all_set
        # non-empty description
        return set.intersection(*[sel.representation for sel in self._selectors])

    @property
    def size_sg(self):
        return len(self.representation)

    def append_and(self, to_append):
        super().append_and(to_append)
        self.representation = self.representation.intersection(to_append.representation)
        self.arr_for_interface = np.array(list(self.representation), dtype=int)

    @property
    def __array_interface__(self):
        return self.arr_for_interface.__array_interface__  # pylint: disable=no-member


class SetRepresentation(RepresentationBase):
    Conjunction = Set_Conjunction

    def __init__(self, df, selectors_to_patch):
        self.df = df
        super().__init__(Set_Conjunction, selectors_to_patch)

    def patch_selector(self, sel):
        sel.representation = set(*np.nonzero(sel.covers(self.df)))
        sel.size_sg = len(sel.representation)

    def patch_classes(self):
        Set_Conjunction.all_set = set(self.df.index)
        super().patch_classes()


class NumpySet_Conjunction(Conjunction):
    all_set = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation = self.compute_representation()

    def compute_representation(self):
        # empty description ==> return a list of all '1's
        if not self._selectors:
            return NumpySet_Conjunction.all_set
        start = self._selectors[0].representation
        for sel in self._selectors[1:]:
            start = np.intersect1d(start, sel.representation, assume_unique=True)
        return start

    @property
    def size_sg(self):
        return len(self.representation)

    def append_and(self, to_append):
        super().append_and(to_append)
        # self._selectors.append(to_append)
        self.representation = np.intersect1d(
            self.representation, to_append.representation, True
        )

    @property
    def __array_interface__(self):
        return self.representation.__array_interface__


class NumpySetRepresentation(RepresentationBase):
    Conjunction = NumpySet_Conjunction

    def __init__(self, df, selectors_to_patch):
        self.df = df
        super().__init__(NumpySet_Conjunction, selectors_to_patch)

    def patch_selector(self, sel):
        sel.representation = np.nonzero(sel.covers(self.df))[0]
        sel.size_sg = len(sel.representation)

    def patch_classes(self):
        NumpySet_Conjunction.all_set = np.arange(len(self.df))
        super().patch_classes()
