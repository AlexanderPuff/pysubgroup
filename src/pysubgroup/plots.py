import matplotlib.pyplot as plt
import numpy as np

path = '/home/alexpuff/runtimes/runtimes.txt'
times=[]


def time_to_float(time_str):
    hours, minutes, seconds = time_str.split(':')
    seconds, microseconds = seconds.split('.')
    
    total_seconds = (
        int(hours) * 3600 +
        int(minutes) * 60 +
        int(seconds) +
        int(microseconds) / 1000000
    )
    
    return total_seconds
           
with open(path) as file:
    for line in file:
        if line[0]=='0':
            seconds = time_to_float(line)
            times.append(seconds/5)
        else:
            times.append(-1)
times.append(-1)
sublists = []
cur= []
for time in times:
    if time ==-1:
        if cur:
            sublists.append(cur)
            cur=[]
    else:
        cur.append(time)
        

cpu_and =sublists[0]
cpu_cnt = sublists[1]
ti2080_and = sublists[2]
ti2080_cnt = sublists[3]
a6000_and = sublists[4]
a6000_cnt = sublists[5][:-1]
lengths = list(2**n for n in range(max(len(lst) for lst in sublists)))

def semilogand():
    plt.figure(figsize=(10, 6))
    plt.semilogx(lengths[:len(cpu_and)], cpu_and, label='Logical and, CPU')
    plt.semilogx(lengths[:len(ti2080_and)], ti2080_and, label='Logical and, 2080ti')
    plt.semilogx(lengths[:len(a6000_and)], a6000_and, label='Logical and, A6000')

    plt.xlabel('Length')
    plt.ylabel('Time (s)')
    plt.title('Log Plot of CPU and GPU Runtimes for logical_and')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('/home/alexpuff/runtimes/newAndSemiLog.pdf')
    
def semilogcnt():
    plt.figure(figsize=(10, 6))
    plt.semilogx(lengths[:len(cpu_cnt)], cpu_cnt, label='count_nonzero, CPU')
    plt.semilogx(lengths[:len(ti2080_cnt)], ti2080_cnt, label='count_nonzero, 2080ti')
    plt.semilogx(lengths[:len(a6000_cnt)], a6000_cnt, label='count_nonzero, A6000')

    plt.xlabel('Length')
    plt.ylabel('Time (s)')
    plt.title('Log Plot of CPU and GPU Runtimes for count_nonzero')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('/home/alexpuff/runtimes/newCntSemiLog.pdf')
    
def loglogand():
    plt.figure(figsize=(10, 6))
    plt.loglog(lengths[:len(cpu_and)], cpu_and, label='CPU')
    plt.loglog(lengths[:len(ti2080_and)], ti2080_and, label='2080ti')
    plt.loglog(lengths[:len(a6000_and)], a6000_and, label='A6000')

    plt.xlabel('Array Length')
    plt.ylabel('Time (s)')
    plt.title('logical_and')
    plt.legend()
    plt.grid(True, which="major", ls="-")
    plt.savefig('/home/alexpuff/runtimes/newAndLogLog.pdf')
    
def loglogcnt():
    plt.figure(figsize=(10, 6))
    plt.loglog(lengths[:len(cpu_cnt)], cpu_cnt, label='CPU')
    plt.loglog(lengths[:len(ti2080_cnt)], ti2080_cnt, label='2080ti')
    plt.loglog(lengths[:len(a6000_cnt)], a6000_cnt, label='A6000')

    plt.xlabel('Array Length')
    plt.ylabel('Time (s)')
    plt.title('count_nonzero')
    plt.legend()
    plt.grid(True, which="major", ls="-")
    plt.savefig('/home/alexpuff/runtimes/newCntLogLog.pdf')

def both():
    plt.figure(figsize=(10, 6))
    plt.loglog(lengths[:len(cpu_cnt)], cpu_cnt, label='CPU, COUNT', color='blue')
    plt.loglog(lengths[:len(cpu_and)], cpu_and, label='CPU, AND' ,color='blue', ls='--')
    plt.loglog(lengths[:len(ti2080_cnt)], ti2080_cnt, label='2080ti, COUNT', color='green')
    plt.loglog(lengths[:len(ti2080_and)], ti2080_and, label='2080ti, AND', color='green', ls='--')
    plt.loglog(lengths[:len(a6000_cnt)], a6000_cnt, label='A6000, COUNT', color='orange')
    plt.loglog(lengths[:len(a6000_and)], a6000_and, label='A6000, AND', color='orange', ls='--')

    plt.xlabel('Array Length')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True, which="major", ls="-")
    plt.savefig('/home/alexpuff/runtimes/bothLogLog.pdf')

both()