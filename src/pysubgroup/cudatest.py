if (False):    
    import timeit

    def time():

        x="""np.logical_and(a,b)"""
        y="""cp.logical_and(a,b)"""
        xs="""
        import numpy as np
        l=100000000
        a=np.random.choice([True, False], size=l)
        b=np.random.choice([True, False], size=l)
        """
        ys="""
        import cupy as cp
        l=100000000
        a=cp.random.choice([True, False], size=l)
        b=cp.random.choice([True, False], size=l)
        """

        npt=timeit.timeit(x, xs, number=10)
        cpt=timeit.timeit(y, ys, number=10)

        print(f"numpy: {npt/10}, cupy: {cpt/10}")

    import cupy as cp
    def cnt(arr):
        return cp.count_nonzero(arr)
    arr=cp.array([True,False,True])
    print(cnt(arr))

from cupyx.profiler import benchmark as bm
import numpy as np
import cupy as cp
import json

def log_runtimes_to_file(file_path, results):
    with open(file_path, 'r') as file:
        try:
            data=json.load(file)
        except json.JSONDecodeError:
            data=[]
    data.extend(results)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
        
cpuCnt= '/home/alexpuff/runtimes/cpuCntNonzero.json'
gpuCnt= '/home/alexpuff/runtimes/gpuCntNonzero.json'
cpuAnd= '/home/alexpuff/runtimes/cpuAnd.json'
gpuAnd= '/home/alexpuff/runtimes/gpuAnd.json'

cpuCntS= '/home/alexpuff/runtimes/cpuCntNonzeroStats.json'
gpuCntS= '/home/alexpuff/runtimes/gpuCntNonzeroStats.json'
cpuAndS= '/home/alexpuff/runtimes/cpuAndStats.json'
gpuAndS= '/home/alexpuff/runtimes/gpuAndStats.json'


def measure_runtimes():
    with(open(cpuCnt, 'w')) as file:
        json.dump([],file)
    with(open(gpuCnt, 'w')) as file:
        json.dump([],file)
    with(open(cpuAnd, 'w')) as file:
        json.dump([],file)
    with(open(gpuAnd, 'w')) as file:
        json.dump([],file)
    n=1
    for i in range(31):
        a=np.random.choice([True, False], size=n)
        b=np.random.choice([True, False], size=n)
        c=cp.random.choice([True, False], size=n)
        d=cp.random.choice([True, False], size=n)
        results=[]
        runtimes=bm(np.count_nonzero, (a,), n_repeat=10)
        results.append({
            "length": n,
            "CPU": runtimes.cpu_times.tolist(),
            "GPU": runtimes.gpu_times.tolist()[0]
        })
        log_runtimes_to_file(cpuCnt ,results)
        
        results=[]
        runtimes=bm(cp.count_nonzero, (c,), n_repeat=10)
        results.append({
            "length": n,
            "CPU": runtimes.cpu_times.tolist(),
            "GPU": runtimes.gpu_times.tolist()[0]
        })
        log_runtimes_to_file(gpuCnt ,results)
        
        results=[]
        runtimes=bm(np.logical_and, (a,b), n_repeat=10)
        results.append({
            "length": n,
            "CPU": runtimes.cpu_times.tolist(),
            "GPU": runtimes.gpu_times.tolist()[0]
        })
        log_runtimes_to_file(cpuAnd ,results)
        
        results=[]
        runtimes=bm(cp.logical_and, (c,d), n_repeat=10)
        results.append({
            "length": n,
            "CPU": runtimes.cpu_times.tolist(),
            "GPU": runtimes.gpu_times.tolist()[0]
        })
        log_runtimes_to_file(gpuAnd ,results)
        n=n*2

def calc_stats():
    times=[cpuCnt, cpuAnd, gpuCnt, gpuAnd]
    stats=[cpuCntS, cpuAndS, gpuCntS, gpuAndS]
    for i in range(4):
        path = times[i]
        pathS = stats[i]
        with open(pathS, 'w') as file:
            json.dump({}, file)
        with open(path, 'r') as file:
            data = json.load(file)
            
        lengths = [entry['length'] for entry in data]       
        cpu_runtimes = [entry['CPU'] for entry in data]
        gpu_runtimes = [entry['GPU'] for entry in data]
        
        results = []
        for i, length in enumerate(lengths):
            cpu_avg = np.mean(cpu_runtimes[i])
            cpu_min = np.min(cpu_runtimes[i])
            cpu_max = np.max(cpu_runtimes[i])
            
            gpu_avg = np.mean(gpu_runtimes[i])
            gpu_min = np.min(gpu_runtimes[i])
            gpu_max = np.max(gpu_runtimes[i])
            
            results.append({
                "length": length,
                "CPU_avg": cpu_avg,
                "CPU_min": cpu_min,
                "CPU_max": cpu_max,
                "GPU_avg": gpu_avg,
                "GPU_min": gpu_min,
                "GPU_max": gpu_max
            })
        with open(pathS, 'w') as file:
            json.dump(results, file, indent=4)
            
