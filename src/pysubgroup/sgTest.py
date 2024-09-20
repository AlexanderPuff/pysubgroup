from io import StringIO
import pandas as pd
import pkg_resources
import pysubgroup as ps
import numpy as np
from cupyx.profiler import benchmark as bm
import cudf
import datetime



def get_data(filepath):
    s_io = StringIO(
        str(pkg_resources.resource_string("pysubgroup", filepath), "utf-8")
    )
    return pd.read_csv(s_io, sep="\t", header=0, nrows=1000000)

def get_data_gpu(filepath):
    s_io = StringIO(
        str(pkg_resources.resource_string("pysubgroup", filepath), "utf-8")
    )
    return cudf.read_csv(s_io, sep="\t", header=0, nrows=1000000)

synth="data/synthetic/synth_titanic.csv"
normal="data/titanic.csv"

gpu_data= get_data_gpu(synth)
cpu_data=get_data(synth)
a=datetime.datetime.now()
searchspace=ps.create_selectors(cpu_data, ignore=['Survived'])
print(datetime.datetime.now()-a)


def measureCPU(data):
    target= ps.BinaryTarget('Survived', True)
    searchspace=ps.create_selectors(data, ignore=['Survived'])
    task = ps.SubgroupDiscoveryTask (
            data,
            target,
            searchspace,
            result_set_size=10,
            depth=2,
            qf=ps.WRAccQF())
    a = datetime.datetime.now()
    result=ps.DFS(ps.BitSetRepresentation).execute(task)
    b = datetime.datetime.now()
    print(b-a)
    
def measureGPU(data):
    target= ps.BinaryTarget('Survived', True)
    searchspace=ps.create_selectors(data, ignore=['Survived'])
    task = ps.SubgroupDiscoveryTask (
            data,
            target,
            searchspace,
            result_set_size=10,
            depth=2,
            qf=ps.WRAccQF())
    a = datetime.datetime.now()
    result=ps.DFS(ps.CUDABitSetRepr).execute(task)
    b = datetime.datetime.now()
    print(b-a)


if(False):
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