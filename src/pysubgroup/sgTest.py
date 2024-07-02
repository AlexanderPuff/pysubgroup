import pysubgroup as ps
import numpy as np
from cupyx.profiler import benchmark as bm
import cudf
import time

# Load the example dataset
from pysubgroup.datasets import get_titanic_data, get_titanic_gpu
data = get_titanic_data()
dataGPU = get_titanic_gpu()


target = ps.BinaryTarget ('Survived', True)
searchspace = ps.create_selectors(data, ignore=['Survived'])
task = ps.SubgroupDiscoveryTask (
    data,
    target,
    searchspace,
    result_set_size=10,
    depth=3,
    qf=ps.WRAccQF())
result=ps.DFS(ps.BitSetRepresentation).execute(task)

print(result.to_dataframe())
if False:
    gputask = ps.SubgroupDiscoveryTask (
        dataGPU,
        target,
        searchspace,
        result_set_size=5,
        depth=2,
        qf=ps.WRAccQF())
    print(bm(ps.DFS(ps.CUDABitSetRepr).execute, (gputask,), n_repeat=5))
    print(bm(ps.DFS().execute, (task,), n_repeat=5))