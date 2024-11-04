import cupy as cp
import cudf
from collections import namedtuple


class GpuSearchSpace:

    def __init__(
        self, data, target_attribute, target_low, target_high=None, nbins=5, ignore=[]
    ):

        self.data = data
        self.instances = data.shape[0]

        # Store selectors in dataframe for easy access to their attributes, starting with the target
        self.sels = cudf.DataFrame(
            {
                "id": 0,
                "attribute": target_attribute,
                "low": str(target_low),
                "high": str(target_high),
                "type": "Target",
            }
        )
        self.target_repr = self.compute_target_repr(
            target_attribute, target_low, target_high
        )
        self.positives = float(cp.count_nonzero(self.target_repr))
        self.sel_id = 1

        # Initialize all other selectors
        self.create_selectors_on_GPU(nbins, ignore)
        self.sels = self.sels.set_index("id", drop=False)

        # Cache selectors' stats for later
        self.stats = StatisticsGpu(self)
        self.sel_stats = self.stats.compute_stats_sels()

        # Don't need original data anymore
        del self.data

    def compute_target_repr(self, target_attribute, target_low, target_high):

        # If only one target value specified, treat as equality selector
        if target_high == None:
            return cp.from_dlpack(
                (self.data[target_attribute] == target_low).to_dlpack()
            )

        # Otherwise as interval:
        else:
            lo = cp.from_dlpack((self.data[target_attribute] >= target_low).to_dlpack())
            hi = cp.from_dlpack((self.data[target_attribute] < target_high).to_dlpack())
            return cp.logical_and(lo, hi)

    def create_selectors_on_GPU(self, nbins=5, ignore=[]):

        self.create_numeric_gpu(nbins, ignore)
        self.create_categorical_gpu(ignore)

    def create_numeric_gpu(self, nbins, ignore):

        # Pre-compute cut points
        indices = cp.arange(0, self.data.shape[0], self.data.shape[0] // nbins + 1)
        # These lists are concatenated later in one go
        selectors = [self.sels]
        representations = [self.target_repr.reshape(-1, 1)]

        for attribute in [
            col
            for col in self.data.select_dtypes(include=["number"])
            if col not in ignore
        ]:
            uniques = cp.from_dlpack(self.data[attribute].unique().dropna().to_dlpack())

            # single value column, no selectors needed
            if len(uniques) == 1:
                continue

            # create categorical selector
            elif len(uniques) <= nbins:
                selectors.append(
                    cudf.DataFrame(
                        {
                            "id": cp.arange(
                                self.sel_id, self.sel_id + len(uniques), dtype=cp.uint16
                            ),
                            "attribute": attribute,
                            "low": arr_to_strings(uniques),
                            "high": None,
                            "type": "categorical",
                        }
                    )
                )

                # sel_id ticks up alongside the number of selectors so far, used to uniquely identify them later
                self.sel_id += len(uniques)

                # Reshape to perpendicular axes for elementwise comparison
                data = cp.from_dlpack(self.data[attribute].to_dlpack()).reshape(-1, 1)
                uniques = uniques.reshape(1, -1)
                representations.append(data == uniques)

            # Create interval selectors
            else:
                sorted_data = cp.sort(
                    cp.from_dlpack((self.data[attribute]).to_dlpack())
                )
                maximum = sorted_data[-1] + 1
                values = cp.unique(sorted_data.take(indices))

                # Sometimes there are no cutpoints, for example when almost all values are bundled up around an extreme
                if len(values) > 1:
                    # Int wouldn't work with infinity
                    if values.dtype == cp.int_:
                        values = cp.append(values, maximum)
                    else:
                        values = cp.append(values, cp.inf)
                    selectors.append(
                        cudf.DataFrame(
                            {
                                "id": cp.arange(
                                    self.sel_id,
                                    self.sel_id + len(values) - 1,
                                    dtype=cp.uint16,
                                ),
                                "attribute": attribute,
                                "low": arr_to_strings(values[:-1]),
                                "high": arr_to_strings(values[1:]),
                                "type": "interval",
                            }
                        )
                    )
                    self.sel_id += len(values) - 1
                    data = cp.from_dlpack(self.data[attribute].to_dlpack()).reshape(
                        -1, 1
                    )
                    values = values.reshape(1, -1)
                    representations.append(
                        cp.logical_and(data >= values[:, :-1], data < values[:, 1:])
                    )

        # Concatenate selectors and representations into one Dataframe and array
        self.sels = cudf.concat(selectors)
        self.reps = cp.swapaxes(cp.hstack(representations), 0, 1)

    def create_categorical_gpu(self, ignore):
        # Again, use list for one concatenation later
        selectors = [self.sels]
        representations = [self.reps]

        for attribute in [
            col
            for col in self.data.select_dtypes(exclude=["number"])
            if col not in ignore
        ]:
            uniques = self.data[attribute].unique().dropna()
            if len(uniques) > 1:

                selectors.append(
                    cudf.DataFrame(
                        {
                            "id": cp.arange(
                                self.sel_id, self.sel_id + len(uniques), dtype=cp.uint16
                            ),
                            "attribute": attribute,
                            "low": arr_to_strings(uniques),
                            "high": None,
                            "type": "categorical",
                        }
                    )
                )
                self.sel_id += len(uniques)
                # iterate here, cupy does not support strings
                for unique in uniques.to_numpy():
                    representations.append(self.data[attribute] == unique)

        self.sels = cudf.concat(selectors)
        self.reps = cp.vstack(representations)


# This class is responsible for computing statistics and quality measures
class StatisticsGpu:

    def __init__(self, search_space):
        ConstantStats = namedtuple(
            "ConstantStats",
            [
                "size_sg",
                "size_dataset",
                "positives_sg",
                "positives_dataset",
                "size_complement",
                "relative_size_sg",
                "coverage_sg",
                "relative_size_complement",
                "coverage_complement",
                "target_share_sg",
                "target_share_dataset",
                "target_share_complement",
                "lift",
            ],
        )

        self.search_space = search_space

        # Pre-compute constant stats for entire dataset
        size = self.search_space.data.shape[0]
        pos = self.search_space.positives
        tshare = pos / size

        self.constant_stats = ConstantStats(
            size_sg=size,
            size_dataset=size,
            positives_sg=pos,
            positives_dataset=pos,
            size_complement=0,
            relative_size_sg=1,
            coverage_sg=1,
            relative_size_complement=0,
            coverage_complement=0,
            target_share_sg=tshare,
            target_share_dataset=tshare,
            target_share_complement=float("nan"),
            lift=1,
        )

    # Computes all statistics for a subgroup based on size and number of positives
    def compute_stats(self, counts, positives):
        statistics = cudf.DataFrame({"size_sg": counts, "positives_sg": positives})
        statistics["size_complement"] = (-statistics["size_sg"]).add(
            self.constant_stats.size_dataset
        )
        statistics["relative_size_sg"] = statistics["size_sg"].truediv(
            self.constant_stats.size_dataset
        )
        statistics["relative_size_complement"] = (-statistics["relative_size_sg"]).add(
            1
        )
        statistics["coverage_sg"] = statistics["positives_sg"].truediv(
            self.constant_stats.positives_dataset
        )
        statistics["coverage_complement"] = 1 - statistics["coverage_sg"]
        statistics["target_share_sg"] = (
            statistics["positives_sg"] / statistics["size_sg"]
        )
        statistics["target_share_complement"] = (
            (-statistics["positives_sg"]).add(self.constant_stats.positives_dataset)
        ) / statistics["size_complement"]
        statistics["lift"] = statistics["target_share_sg"].truediv(
            self.constant_stats.target_share_dataset
        )
        return statistics

    def compute_quality(self, statistics, a):
        return statistics["relative_size_sg"].pow(a) * (
            statistics["target_share_sg"].add(-self.constant_stats.target_share_dataset)
        )

    def compute_optimistic(self, statistics, a):
        return (
            (statistics["positives_sg"].truediv(self.constant_stats.size_dataset))
            .pow(a)
            .multiply(1 - self.constant_stats.target_share_dataset)
        )

    def compute_stats_sels(self):
        # Count size and positives of all subgroups defined by one selector
        reps = self.search_space.reps
        cnt = cp.count_nonzero(reps, axis=1)
        pos = cp.count_nonzero(reps & reps[0], axis=1)

        # Compute full statistics for them
        statistics = self.compute_stats(cnt, pos)
        return statistics

    def compute_quality_optimistic(self, sgs, a):
        # Every sg here should have same depth
        depth = cp.sum(sgs[0] != 0).item()

        # Start with reps of only first selectors, add the others iteratively
        reps = self.search_space.reps[sgs[:, 0]]
        for d in range(1, depth):
            reps = reps & self.search_space.reps[sgs[:, d]]

        # Count resulting reps
        cnt = cp.sum(reps, axis=1)
        pos = cp.sum(reps & self.search_space.reps[0], axis=1)

        stats = cudf.DataFrame()
        stats["size_sg"] = cnt
        stats["positives_sg"] = pos
        stats["relative_size_sg"] = stats["size_sg"].truediv(
            self.constant_stats.size_dataset
        )
        stats["target_share_sg"] = stats["positives_sg"] / stats["size_sg"]

        q = self.compute_quality(stats, a)
        o = self.compute_optimistic(stats, a)
        return q, o

    def get_cover_arr_sels(self, sels):
        # AND of all selectors in sels
        cover_arr = cp.all(self.search_space.reps[sels], axis=0)
        return cover_arr


def arr_to_strings(arr):
    # Helper function converting cupy array or cudf series into a list of strings
    if type(arr) == cudf.Series:
        arr = arr.to_numpy()
    else:
        arr = cp.asnumpy(arr)
    if arr.dtype == float:
        arr = [round(x, 3) for x in arr]
    strings = [str(x) for x in arr]
    return strings
