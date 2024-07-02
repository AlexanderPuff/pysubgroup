import json
import matplotlib.pyplot as plt
import numpy as np

def andLogLog():
    with open('/home/alexpuff/runtimes/cpuAndStats.json', 'r') as cpu:
        cpuData = json.load(cpu)
        
    with open('/home/alexpuff/runtimes/gpuAndStats.json', 'r') as gpu:
        gpuData = json.load(gpu)
        
    lengths = [entry['length'] for entry in cpuData]
    np_cpu_avg = [entry['CPU_avg'] for entry in cpuData]
    np_gpu_avg = [entry['GPU_avg'] for entry in cpuData]
    cp_cpu_avg = [entry['CPU_avg'] for entry in gpuData]
    cp_gpu_avg = [entry['GPU_avg'] for entry in gpuData]

    np_avg = np.maximum(np_cpu_avg, np_gpu_avg)
    cp_avg = np.maximum(cp_cpu_avg, cp_gpu_avg)

    plt.figure(figsize=(10, 6))
    plt.loglog(lengths, np_avg, label='Numpy: logical and', marker='o')
    plt.loglog(lengths, cp_avg, label='Cupy: logical and', marker='o')


    plt.xlabel('Length')
    plt.ylabel('Time (s)')
    plt.title('Log-Log Plot of CPU and GPU Runtimes for logical_and')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Save the plot as PDF
    plt.savefig('/home/alexpuff/runtimes/andLogLog.pdf')
    
def andSemiLog():
    with open('/home/alexpuff/runtimes/cpuAndStats.json', 'r') as cpu:
        cpuData = json.load(cpu)
        
    with open('/home/alexpuff/runtimes/gpuAndStats.json', 'r') as gpu:
        gpuData = json.load(gpu)
        
    lengths = [entry['length'] for entry in cpuData]
    np_cpu_avg = [entry['CPU_avg'] for entry in cpuData]
    np_gpu_avg = [entry['GPU_avg'] for entry in cpuData]
    cp_cpu_avg = [entry['CPU_avg'] for entry in gpuData]
    cp_gpu_avg = [entry['GPU_avg'] for entry in gpuData]

    np_avg = np.maximum(np_cpu_avg, np_gpu_avg)
    cp_avg = np.maximum(cp_cpu_avg, cp_gpu_avg)

    plt.figure(figsize=(10, 6))
    plt.semilogx(lengths, np_avg, label='Numpy: logical and', marker='o')
    plt.semilogx(lengths, cp_avg, label='Cupy: logical and', marker='o')


    plt.xlabel('Length')
    plt.ylabel('Time (s)')
    plt.title('Log Plot of CPU and GPU Runtimes for logical_and')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Save the plot as PDF
    plt.savefig('/home/alexpuff/runtimes/andSemiLog.pdf')
    
def cntLogLog():
    with open('/home/alexpuff/runtimes/cpuCntNonzeroStats.json', 'r') as cpu:
        cpuData = json.load(cpu)
        
    with open('/home/alexpuff/runtimes/gpuCntNonzeroStats.json', 'r') as gpu:
        gpuData = json.load(gpu)
        
    lengths = [entry['length'] for entry in cpuData]
    np_cpu_avg = [entry['CPU_avg'] for entry in cpuData]
    np_gpu_avg = [entry['GPU_avg'] for entry in cpuData]
    cp_cpu_avg = [entry['CPU_avg'] for entry in gpuData]
    cp_gpu_avg = [entry['GPU_avg'] for entry in gpuData]

    np_avg = np.maximum(np_cpu_avg, np_gpu_avg)
    cp_avg = np.maximum(cp_cpu_avg, cp_gpu_avg)

    plt.figure(figsize=(10, 6))
    plt.loglog(lengths, np_avg, label='Numpy: count nonzero', marker='o')
    plt.loglog(lengths, cp_avg, label='Cupy: count nonzero', marker='o')


    plt.xlabel('Length')
    plt.ylabel('Time (s)')
    plt.title('Log-Log Plot of CPU and GPU Runtimes for count_nonzero')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Save the plot as PDF
    plt.savefig('/home/alexpuff/runtimes/countLogLog.pdf')
    
def cntSemiLog():
    with open('/home/alexpuff/runtimes/cpuCntNonzeroStats.json', 'r') as cpu:
        cpuData = json.load(cpu)
        
    with open('/home/alexpuff/runtimes/gpuCntNonzeroStats.json', 'r') as gpu:
        gpuData = json.load(gpu)
        
    lengths = [entry['length'] for entry in cpuData]
    np_cpu_avg = [entry['CPU_avg'] for entry in cpuData]
    np_gpu_avg = [entry['GPU_avg'] for entry in cpuData]
    cp_cpu_avg = [entry['CPU_avg'] for entry in gpuData]
    cp_gpu_avg = [entry['GPU_avg'] for entry in gpuData]

    np_avg = np.maximum(np_cpu_avg, np_gpu_avg)
    cp_avg = np.maximum(cp_cpu_avg, cp_gpu_avg)

    plt.figure(figsize=(10, 6))
    plt.semilogx(lengths, np_avg, label='Numpy: logial and', marker='o')
    plt.semilogx(lengths, cp_avg, label='Cupy: logical and', marker='o')


    plt.xlabel('Length')
    plt.ylabel('Time (s)')
    plt.title('Log Plot of CPU and GPU Runtimes for count_nonzero')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Save the plot as PDF
    plt.savefig('/home/alexpuff/runtimes/countSemiLog.pdf')
    
andLogLog()
andSemiLog()
cntLogLog()
cntSemiLog()
