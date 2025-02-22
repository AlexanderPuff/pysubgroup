"""
Created on 28.04.2016

@author: lemmerfn
"""
import copy
import weakref
from abc import ABC, abstractmethod
from functools import total_ordering
from itertools import chain
import cudf
import cupy as cp
import numpy as np
import pysubgroup as ps


@total_ordering
class SelectorBase(ABC):
    # selector cache
    __refs__ = weakref.WeakSet()

    def __new__(cls, *args, **kwargs):
        """Ensures that each selector only exists once."""

        # create temporary selector
        tmp = super().__new__(cls)
        tmp.set_descriptions(*args, **kwargs)

        # save original arguments
        # NOTE: this is a fix for pickle
        #       so we can call `__getnewargs_ex__` with the right arguments
        # TODO: this may have unintended side effects if args,
        #       kwargs are large or volatile (I don't think we have that yet though)
        tmp.__new_args__ = args, kwargs

        # check if selector is already in cache (__refs__)
        # if so, return cached instance
        if tmp not in SelectorBase.__refs__:
            return tmp  # new selector
        # (not sure why we never have the below case)
        for ref in SelectorBase.__refs__:  # pragma no branch okay
            if ref == tmp:
                return ref
        # if not return
        return tmp  # pragma: no cover

    def __getnewargs_ex__(self):  # pylint: disable=invalid-getnewargs-ex-returned
        tmp_args = self.__new_args__
        del self.__new_args__
        return tmp_args

    def __init__(self):
        # add selector to cache
        # TODO: why not do this in `__new__`,
        # then it would be all together in one function?
        SelectorBase.__refs__.add(self)

    def __eq__(self, other):
        if other is None:  # pragma: no cover
            return False
        return repr(self) == repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __hash__(self):
        return self._hash  # pylint: disable=no-member

    @abstractmethod
    def set_descriptions(self, *args, **kwargs):
        pass  # pragma: no cover


def get_cover_array_and_size(subgroup, data_len=None, data=None):
    if hasattr(subgroup, "representation"):
        cover_arr = subgroup
        size = subgroup.size_sg
    elif isinstance(subgroup, slice):
        cover_arr = subgroup
        if data_len is None:
            if type(data).__name__ == "DataFrame":
                data_len = len(data)
            else:
                raise ValueError(
                    "if you pass a slice, you need to pass either data_len or data"
                )
        # https://stackoverflow.com/questions/36188429/retrieve-length-of-slice-from-slice-object-in-python
        size = len(range(*subgroup.indices(data_len)))
    elif hasattr(subgroup, "__array_interface__"):
        cover_arr = subgroup
        type_char = subgroup.__array_interface__["typestr"][1]
        if type_char == "b":  # boolean indexing is used
            size = np.count_nonzero(cover_arr)
        elif type_char in ("u", "i"):  # integer indexing
            size = subgroup.__array_interface__["shape"][0]
        else:
            raise NotImplementedError(
                f"Currently a typechar of {type_char} is not supported."
            )
    else:
        assert type(data).__name__ == "DataFrame", str(type(data))
        cover_arr = subgroup.covers(data)
        size = np.count_nonzero(cover_arr)
    return cover_arr, size


def get_size(subgroup, data_len=None, data=None):
    if hasattr(subgroup, "representation"):
        size = subgroup.size_sg
    elif isinstance(subgroup, slice):
        if data_len is None:
            if type(data).__name__ == "DataFrame":
                data_len = len(data)
            else:
                raise ValueError(
                    "if you pass a slice, you need to pass either data_len or data"
                )
        # https://stackoverflow.com/questions/36188429/retrieve-length-of-slice-from-slice-object-in-python
        size = len(range(*subgroup.indices(data_len)))
    elif hasattr(subgroup, "__array_interface__"):
        type_char = subgroup.__array_interface__["typestr"][1]
        if type_char == "b":  # boolean indexing is used
            size = np.count_nonzero(subgroup)
        elif type_char == "u" or type_char == "i":  # integer indexing
            size = subgroup.__array_interface__["shape"][0]
        else:
            raise NotImplementedError(
                f"Currently a typechar of {type_char} is not supported."
            )
    else:
        assert type(data).__name__ == "DataFrame"
        size = np.count_nonzero(subgroup.covers(data))
    return size


def pandas_sparse_eq(col, value):
    import pandas as pd  # pylint: disable=import-outside-toplevel
    from pandas._libs.sparse import (
        IntIndex,  # pylint: disable=import-outside-toplevel, no-name-in-module
    )

    col_arr = col.array
    is_same_value = col_arr.sp_values == value
    new_index_arr = col_arr.sp_index.indices[is_same_value]
    index = IntIndex(len(col), new_index_arr)
    return pd.arrays.SparseArray(
        np.ones(len(new_index_arr), dtype=bool),
        index,
        col_arr.fill_value == value,
        dtype=bool,
    )


class EqualitySelector(SelectorBase):
    def __init__(self, attribute_name, attribute_value, selector_name=None):
        if attribute_name is None:
            raise TypeError()
        if attribute_value is None:
            raise TypeError()

        # TODO: this is redundant due to `__new__` and `set_descriptions`
        self._attribute_name = attribute_name
        self._selector_name = selector_name
        self._attribute_value = attribute_value
        self.set_descriptions(
        self._attribute_name, self._attribute_value, self._selector_name
        )

        super().__init__()

    @property
    def attribute_name(self):
        return self._attribute_name

    @property
    def attribute_value(self):
        return self._attribute_value

    def set_descriptions(
        self, attribute_name, attribute_value, selector_name=None
    ):  # pylint: disable=arguments-differ
        self._hash, self._query, self._string = EqualitySelector.compute_descriptions(
            attribute_name, attribute_value, selector_name=selector_name
        )

    @classmethod
    def compute_descriptions(cls, attribute_name, attribute_value, selector_name):
        if isinstance(attribute_value, (str, bytes)):
            query = str(attribute_name) + "==" + "'" + str(attribute_value) + "'"
        elif attribute_value is None:
            query = str(attribute_name) + " is None"
        elif np.isnan(attribute_value):
            query = attribute_name + ".isnull()"
        else:
            query = str(attribute_name) + "==" + str(attribute_value)
        if selector_name is not None:
            string_ = selector_name
        else:
            string_ = query
        hash_value = hash(query)
        return (hash_value, query, string_)

    def __repr__(self):
        return self._query

    def covers(self, data):
        if isinstance(data, cudf.DataFrame):
            column = data[self.attribute_name]
            if column.dtype != object:
                column = cp.fromDlpack(column.to_dlpack())
                return (column == self.attribute_value)
            else:
                return cp.fromDlpack((column == self.attribute_value).to_dlpack())
        else:
            import pandas as pd  # pylint: disable=import-outside-toplevel

            column = data[self.attribute_name]
            if isinstance(column.dtype, pd.SparseDtype):
                row = column
                if not pd.isnull(self.attribute_value):
                    return pandas_sparse_eq(column, self.attribute_value)
            else:
                row = column.to_numpy()
            if pd.isnull(self.attribute_value):
                return pd.isnull(row)
            return row == self.attribute_value

    
    def __str__(self, open_brackets="", closing_brackets=""):
        return open_brackets + self._string + closing_brackets

    @property
    def selectors(self):
        return (self,)

    @staticmethod
    def from_str(s):
        s = s.strip()
        attribute_name, attribute_value = s.split("==")
        if attribute_value[0] == "'" and attribute_value[-1] == "'":
            if attribute_value.startswith("'b'") and attribute_value.endswith("''"):
                attribute_value = str.encode(attribute_value[3:-2])
            else:
                attribute_value = attribute_value[1:-1]
        try:
            attribute_value = int(attribute_value)
        except ValueError:
            try:
                attribute_value = float(attribute_value)
            except ValueError:
                pass
        return ps.EqualitySelector(attribute_name, attribute_value)


class NegatedSelector(SelectorBase):
    def __init__(self, selector):
        # TODO: this is redundant due to `__new__` and `set_descriptions`
        self._selector = selector
        self.set_descriptions(selector)

        super().__init__()

    def covers(self, data_instance):
        if isinstance(data_instance, cudf.DataFrame):
            return cp.logical_not(self._selector.covers(data_instance))
        else:
            return np.logical_not(self._selector.covers(data_instance))

    def __repr__(self):
        return self._query

    def __str__(self, open_brackets="", closing_brackets=""):
        return "NOT " + self._selector.__str__(open_brackets, closing_brackets)

    def set_descriptions(self, selector):  # pylint: disable=arguments-differ
        self._query = "(not " + repr(selector) + ")"
        self._hash = hash(repr(self))

    @property
    def attribute_name(self):
        return self._selector.attribute_name

    @property
    def selectors(self):
        return (self,)


# Including the lower bound, excluding the upper_bound
class IntervalSelector(SelectorBase):
    def __init__(self, attribute_name, lower_bound, upper_bound, selector_name=None):
        assert lower_bound < upper_bound
        # TODO: this is redundant due to `__new__` and `set_descriptions`
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._attribute_name = attribute_name
        self.selector_name = selector_name
        self.set_descriptions(attribute_name, lower_bound, upper_bound, selector_name)

        super().__init__()

    @property
    def attribute_name(self):
        return self._attribute_name

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    def covers(self, data_instance):
        if isinstance(data_instance, cudf.DataFrame):
            column = cp.fromDlpack(data_instance[self.attribute_name].to_dlpack())
            lower = (column >= self.lower_bound)
            if cp.isinf(self.upper_bound):
                return lower
            upper = (column < self.upper_bound)
            return cp.logical_and(lower, upper)
        else:
            val = data_instance[self.attribute_name].to_numpy()
            return np.logical_and((val >= self.lower_bound), (val < self.upper_bound))
        

    def __repr__(self):
        return self._query

    def __hash__(self):
        return self._hash

    def __str__(self):
        return self._string

    @classmethod
    def compute_descriptions(
        cls, attribute_name, lower_bound, upper_bound, selector_name=None
    ):
        if selector_name is None:
            _string = cls.compute_string(
                attribute_name, lower_bound, upper_bound, rounding_digits=2
            )
        else:
            _string = selector_name
        _query = cls.compute_string(
            attribute_name, lower_bound, upper_bound, rounding_digits=None
        )
        _hash = hash(_query)
        return (_hash, _query, _string)

    def set_descriptions(
        self, attribute_name, lower_bound, upper_bound, selector_name=None
    ):  # pylint: disable=arguments-differ
        self._hash, self._query, self._string = IntervalSelector.compute_descriptions(
            attribute_name, lower_bound, upper_bound, selector_name=selector_name
        )

    @classmethod
    def compute_string(cls, attribute_name, lower_bound, upper_bound, rounding_digits):
        if rounding_digits is None:
            formatter = "{}"
        else:
            formatter = "{0:." + str(rounding_digits) + "f}"
        ub = upper_bound
        lb = lower_bound
        if ub % 1:
            ub = formatter.format(ub)
        if lb % 1:
            lb = formatter.format(lb)

        if lower_bound == float("-inf") and upper_bound == float("inf"):
            repre = str(attribute_name) + " = anything"
        elif lower_bound == float("-inf"):
            repre = str(attribute_name) + "<" + str(ub)
        elif upper_bound == float("inf"):
            repre = str(attribute_name) + ">=" + str(lb)
        else:
            repre = str(attribute_name) + ": [" + str(lb) + ":" + str(ub) + "["
        return repre

    @staticmethod
    def from_str(s):
        s = s.strip()
        if s.endswith(" = anything"):
            return IntervalSelector(
                s[: -len(" = anything")], float("-inf"), float("+inf")
            )
        if "<" in s:
            attribute_name, ub = s.split("<")
            try:
                return IntervalSelector(attribute_name.strip(), float("-inf"), int(ub))
            except ValueError:
                return IntervalSelector(
                    attribute_name.strip(), float("-inf"), float(ub)
                )
        if ">=" in s:
            attribute_name, lb = s.split(">=")
            try:
                return IntervalSelector(attribute_name.strip(), int(lb), float("inf"))
            except ValueError:
                return IntervalSelector(attribute_name.strip(), float(lb), float("inf"))
        if s.count(":") == 2:
            attribute_name, lb, ub = s.split(":")
            lb = lb.strip()[1:]
            ub = ub.strip()[:-1]
            try:
                return IntervalSelector(attribute_name.strip(), int(lb), int(ub))
            except ValueError:
                return IntervalSelector(attribute_name.strip(), float(lb), float(ub))
        else:
            raise ValueError(f"string {s} could not be converted to IntervalSelector")

    @property
    def selectors(self):
        return (self,)


def create_selectors(data, nbins=5, intervals_only=True, ignore=None):
    if ignore is None:
        ignore = []
    sels = create_nominal_selectors(data, ignore)
    sels.extend(create_numeric_selectors(data, nbins, intervals_only, ignore=ignore))
    return sels


def create_nominal_selectors(data, ignore=None):
    if ignore is None:
        ignore = []
    nominal_selectors = []
    # for attr_name in [
    #       x for x in data.select_dtypes(exclude=['number']).columns.values
    #       if x not in ignore]:
    #    nominal_selectors.extend(
    #       create_nominal_selectors_for_attribute(data, attr_name))
    nominal_dtypes = data.select_dtypes(exclude=["number"])
    dtypes = data.dtypes
    for attr_name in [x for x in nominal_dtypes.columns.values if x not in ignore]:
        nominal_selectors.extend(
            create_nominal_selectors_for_attribute(data, attr_name, dtypes)
        )
    return nominal_selectors


def create_nominal_selectors_for_attribute(data, attribute_name, dtypes=None):
    nominal_selectors = []
    if isinstance(data, cudf.DataFrame):
        for val in data[attribute_name].unique().dropna().to_pandas():
            nominal_selectors.append(EqualitySelector(attribute_name, val))
    else:
        import pandas as pd  # pylint: disable=import-outside-toplevel
        for val in pd.unique(data[attribute_name]):
            nominal_selectors.append(EqualitySelector(attribute_name, val))
    # setting the is_bool flag for selector
    if dtypes is None:
        dtypes = data.dtypes
    if dtypes[attribute_name] == "bool":
        for s in nominal_selectors:
            s.is_bool = True
    return nominal_selectors


def create_numeric_selectors(
    data, nbins=5, intervals_only=True, weighting_attribute=None, ignore=None
):
    if ignore is None:
        ignore = []  # pragma: no cover
    numeric_selectors = []
    for attr_name in [
        x
        for x in data.select_dtypes(include=["number"]).columns.values
        if x not in ignore
    ]:
        numeric_selectors.extend(
            create_numeric_selectors_for_attribute(
                data, attr_name, nbins, intervals_only, weighting_attribute
            )
        )
    
    return numeric_selectors


def create_numeric_selectors_for_attribute(
    data, attr_name, nbins=5, intervals_only=True, weighting_attribute=None
):

    numeric_selectors=[]
    if isinstance(data, cudf.DataFrame):
        #can't handle missing values anyways
        data_not_null = data
        uniqueValues = cp.unique(data_not_null[attr_name])
        if len(data_not_null)<len(data):
            numeric_selectors.append(EqualitySelector(attr_name, cp.nan))
    else:
        import pandas as pd  # pylint: disable=import-outside-toplevel

        if isinstance(data[attr_name].dtype, pd.SparseDtype):
            numeric_selectors.append(
                EqualitySelector(attr_name, data[attr_name].sparse.fill_value)
            )
            dense_data = data[attr_name].sparse.sp_values
            data_not_null = dense_data[pd.notnull(dense_data)]
            uniqueValues = np.unique(data_not_null)
            if len(data_not_null) < len(dense_data):
                numeric_selectors.append(EqualitySelector(attr_name, np.nan))
        else:
            data_not_null = data[data[attr_name].notnull()]

            uniqueValues = np.unique(data_not_null[attr_name])
            if len(data_not_null) < len(data):
                numeric_selectors.append(EqualitySelector(attr_name, np.nan))

    if len(uniqueValues) <= nbins:
        for val in uniqueValues:
            numeric_selectors.append(EqualitySelector(attr_name, val))
    else:
        cutpoints = ps.equal_frequency_discretization(
            data, attr_name, nbins, weighting_attribute
        )
        if intervals_only:
            old_cutpoint = float("-inf")
            for c in cutpoints:
                numeric_selectors.append(IntervalSelector(attr_name, old_cutpoint, c))
                old_cutpoint = c
            numeric_selectors.append(
                IntervalSelector(attr_name, old_cutpoint, float("inf"))
            )
        else:
            for c in cutpoints:
                numeric_selectors.append(IntervalSelector(attr_name, c, float("inf")))
                numeric_selectors.append(IntervalSelector(attr_name, float("-inf"), c))

    return numeric_selectors
    
        
def remove_target_attributes(selectors, target):
    return [
        sel for sel in selectors if sel.attribute_name not in target.get_attributes()
    ]


##############
# Boolean expressions
##############
class BooleanExpressionBase(ABC):
    def __or__(self, other):
        tmp = copy.copy(self)
        tmp.append_or(other)
        return tmp

    def __and__(self, other):
        tmp = copy.copy(self)
        tmp.append_and(other)
        return tmp

    @abstractmethod
    def append_and(self, to_append):
        pass

    @abstractmethod
    def append_or(self, to_append):
        pass

    @abstractmethod
    def __copy__(self):
        pass


@total_ordering
class Conjunction(BooleanExpressionBase):
    def __init__(self, selectors):
        self._repr = None
        self._hash = None
        try:
            it = iter(selectors)
            self._selectors = list(it)
        except TypeError:
            self._selectors = [selectors]

    def covers(self, instance):
        # empty description ==> return a list of all '1's
        if isinstance(instance, cudf.DataFrame):
            if not self.selectors:
                return cp.full(len(instance), True, dtype=bool)
            return cp.all(cp.stack([sel.covers(instance) for sel in self._selectors]), axis=0)
        
        if not self._selectors:
            return np.full(len(instance), True, dtype=bool)
        # non-empty description
        return np.all([sel.covers(instance) for sel in self._selectors], axis=0)

    def __len__(self):
        return len(self._selectors)

    def __str__(self, open_brackets="", closing_brackets="", and_term=" AND "):
        if not self._selectors:
            return "Dataset"
        attrs = sorted(str(sel) for sel in self._selectors)
        return "".join((open_brackets, and_term.join(attrs), closing_brackets))

    def __repr__(self):
        if self._repr is not None:
            return self._repr
        else:
            self._repr = self._compute_repr()
            return self._repr

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __hash__(self):
        if self._hash is not None:
            return self._hash
        else:
            self._hash = self._compute_hash()
            return self._hash

    def _compute_repr(self):
        if not self._selectors:
            return "True"
        reprs = sorted(repr(sel) for sel in self._selectors)
        return "(" + " and ".join(reprs) + ")"

    def _compute_hash(self):
        return hash(repr(self))

    def _invalidate_representations(self):
        self._repr = None
        self._hash = None

    def append_and(self, to_append):
        if isinstance(to_append, SelectorBase):
            self._selectors.append(to_append)
        elif isinstance(to_append, Conjunction):
            self._selectors.extend(to_append.selectors)
        else:
            self._selectors.extend(to_append)
        self._invalidate_representations()

    def append_or(self, to_append):
        raise RuntimeError(
            "Or operations are not supported by a pure Conjunction. Consider using DNF."
        )

    def pop_and(self):
        return self._selectors.pop()

    def pop_or(self):
        raise RuntimeError(
            "Or operations are not supported by a pure Conjunction. Consider using DNF."
        )

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result._selectors = list(self._selectors)
        return result

    @property
    def depth(self):
        return len(self._selectors)

    @property
    def selectors(self):
        return tuple(chain.from_iterable(sel.selectors for sel in self._selectors))

    @staticmethod
    def from_str(s):
        if s.strip() == "Dataset":
            return Conjunction([])
        selector_strings = s.split(" AND ")
        selectors = []
        for selector_string in selector_strings:
            selector_string = selector_string.strip()
            if "==" in selector_string:
                selectors.append(EqualitySelector.from_str(selector_string))
            else:
                selectors.append(IntervalSelector.from_str(selector_string))
        return Conjunction(selectors)


@total_ordering
class Disjunction(BooleanExpressionBase):
    def __init__(self, selectors=None):
        if isinstance(selectors, (list, tuple)):
            self._selectors = selectors
        elif selectors is None:
            self._selectors = []
        else:
            self._selectors = [selectors]

    def covers(self, instance):
        # empty description ==> return a list of all '1's
        if not self._selectors:
            return np.full(len(instance), False, dtype=bool)
        # non-empty description
        return np.any([sel.covers(instance) for sel in self._selectors], axis=0)

    def __len__(self):
        return len(self._selectors)

    def __str__(self, open_brackets="", closing_brackets="", or_term=" OR "):
        if not self._selectors:
            return "Empty"  # pragma: no cover
        attrs = sorted(str(sel) for sel in self._selectors)
        return "".join((open_brackets, or_term.join(attrs), closing_brackets))

    def __repr__(self):
        if not self._selectors:
            return "True"
        reprs = sorted(repr(sel) for sel in self._selectors)
        return "".join(("(", " or ".join(reprs), ")"))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __hash__(self):
        return hash(repr(self))

    def append_and(self, to_append):
        raise RuntimeError(
            "And operations are not supported by a pure Conjunction. "
            "Consider using DNF."
        )

    def append_or(self, to_append):
        if isinstance(to_append, Disjunction):
            self._selectors.extend(to_append.selectors)
            return
        try:
            self._selectors.extend(to_append)
        except TypeError:
            self._selectors.append(to_append)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result._selectors = copy.copy(self._selectors)
        return result

    @property
    def selectors(self):
        return tuple(chain.from_iterable(sel.selectors for sel in self._selectors))


class DNF(Disjunction):
    def __init__(self, selectors=None):
        if selectors is None:
            selectors = []
        super().__init__([])
        self.append_or(selectors)

    @staticmethod
    def _ensure_pure_conjunction(to_append):
        if isinstance(to_append, Conjunction):
            return to_append
        elif isinstance(to_append, SelectorBase):
            return Conjunction(to_append)
        else:
            it = iter(to_append)
            if all(isinstance(sel, SelectorBase) for sel in to_append):
                return Conjunction(it)
            else:
                raise ValueError(
                    "DNFs only accept an iterable of Selectors"
                )  # pragma: no cover

    def append_or(self, to_append):
        if isinstance(to_append, ps.Disjunction):
            to_append = to_append.selectors
        try:
            it = iter(to_append)
            conjunctions = [DNF._ensure_pure_conjunction(part) for part in it]
        except TypeError:
            conjunctions = DNF._ensure_pure_conjunction(to_append)
        super().append_or(conjunctions)

    def append_and(self, to_append):
        if isinstance(to_append, Disjunction):
            raise NotImplementedError(
                "Appeding a disjunction to DNF is not implemented"
            )
        conj = DNF._ensure_pure_conjunction(to_append)
        if len(self._selectors) > 0:
            for conjunction in self._selectors:
                conjunction.append_and(conj)
        else:
            self._selectors.append(conj)

    def pop_and(self):
        out_list = [s.pop_and() for s in self._selectors]
        return_val = out_list[0]
        if all(x == return_val for x in out_list):
            return return_val
        else:
            for to_append, conj in zip(out_list, self._selectors):
                conj.append_and(to_append)

            raise RuntimeError("pop_and failed as the result was inconsistent")
