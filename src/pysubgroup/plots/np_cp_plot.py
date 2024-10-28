import matplotlib.pyplot as plt
import pandas as pd

dir = '/home/alexpuff/runtimes/FullTest/'

cpu = 'cpuCleaned.csv'
gpu = 'gpuCleaned.csv'
hor = 'horizontalCleaned.csv'



def get_df(path):
    df = pd.read_csv(path)
    df['Rows'] = df['Rows'].astype(int)
    df['Data Loaded'] = df['Data Loaded'].astype(float)
    df['Search Space'] = df['Search Space'].astype(float)
    df['SG Discovery'] = df['SG Discovery'].astype(float)
    df['Total'] = df['Total'].astype(float)
    return df

def get_avgs(df):
    #result = pd.DataFrame(columns=['Rows', 'Data Loaded', 'Search Space', 'SG Discovery', 'Total'])
    result = []
    for l in pd.unique(df['Rows']):
        times = df[df['Rows']==l][['Data Loaded', 'Search Space', 'SG Discovery', 'Total']]
        times = times.mean(axis=0)
        result.append(times)
    return pd.concat(result, ignore_index=True, axis=1)

cpu_df = get_df(dir+cpu)
gpu_df = get_df(dir+gpu)
hor_df = get_df(dir+hor)

dataset = 'Spam'
plottype = 'semilog'

def filter_df(df, dataset, quality, algorithm, gpu='-'):
    filtered = df[(df['Dataset']==dataset) & (df['Quality']==quality) & (df['Algorithm']==algorithm) & (df['GPU']==gpu)]
    return filtered

#cpu=get_avgs(filter_df(cpu_df, dataset, 1, 'Apriori'))
#gpu_2080=get_avgs(filter_df(gpu_df, dataset, 1, 'Apriori', 'NVIDIA GeForce RTX 2080 Ti'))
#gpu_6000=get_avgs(filter_df(gpu_df, dataset, 1, 'Apriori', 'NVIDIA RTX A6000'))
#hor_2080=get_avgs(filter_df(hor_df, dataset, 1, 'Apriori', 'NVIDIA GeForce RTX 2080 Ti'))
#hor_6000=get_avgs(filter_df(hor_df, dataset, 1, 'Apriori', 'NVIDIA RTX A6000'))

lengths = pd.unique(cpu_df['Rows'])

cpu_a = get_avgs(filter_df(cpu_df, dataset, 1, 'Apriori'))
cpu_d = get_avgs(filter_df(cpu_df, dataset, 1, 'DFS'))
gpu_a = get_avgs(filter_df(gpu_df, dataset, 1, 'Apriori', 'NVIDIA GeForce RTX 2080 Ti'))
gpu_d = get_avgs(filter_df(gpu_df, dataset, 1, 'DFS', 'NVIDIA GeForce RTX 2080 Ti'))
hor_a = get_avgs(filter_df(hor_df, dataset, 1, 'Apriori', 'NVIDIA GeForce RTX 2080 Ti'))
hor_d = get_avgs(filter_df(hor_df, dataset, 1, 'DFS', 'NVIDIA GeForce RTX 2080 Ti'))
#cpu_times = cpu.loc['Search Space'] + cpu.loc['SG Discovery']
#gpu_2080_times = gpu_2080.loc['Search Space'] + gpu_2080.loc['SG Discovery']
#gpu_6000_times = gpu_6000.loc['Search Space'] + gpu_6000.loc['SG Discovery']
#hor_2080_times = hor_2080.loc['Search Space'] + hor_2080.loc['SG Discovery']
#hor_6000_times = hor_6000.loc['Search Space'] + hor_6000.loc['SG Discovery']
cpu_a_times = cpu_a.loc['Search Space'] + cpu_a.loc['SG Discovery']
cpu_d_times = cpu_d.loc['Search Space'] + cpu_d.loc['SG Discovery']
gpu_a_times = gpu_a.loc['Search Space'] + gpu_a.loc['SG Discovery']
gpu_d_times = gpu_d.loc['Search Space'] + gpu_d.loc['SG Discovery']
hor_a_times = hor_a.loc['Search Space'] + hor_a.loc['SG Discovery']
hor_d_times = hor_d.loc['Search Space'] + hor_d.loc['SG Discovery']


if plottype == 'loglog':
    plot = plt.loglog
if plottype == 'semilog':
    plot = plt.semilogx

plt.figure(figsize=(10, 6))
plot(lengths[:len(cpu_a_times)], cpu_a_times, label='Apriori', color='blue')
plot(lengths[:len(cpu_d_times)], cpu_d_times, label='DFS', color='blue',linestyle='--')
plot(lengths[:len(gpu_a_times)], gpu_a_times, label='Apriori GPU', color='orange')
plot(lengths[:len(gpu_d_times)], gpu_d_times, label='DFS GPU', color='orange',linestyle='--')
plot(lengths[:len(hor_a_times)], hor_a_times, label='Apriori Full', color='green')
plot(lengths[:len(hor_d_times)], hor_d_times, label='DFS Full', color='green',linestyle='--')


plt.xlabel('Length')
plt.ylabel('Time (s)')
#plt.title('Log Plot of CPU and GPU Runtimes for count_nonzero')
plt.legend()
plt.grid(True, which="major", ls="--")

# Save the plot as PDF
plt.savefig(dir + 'plots/' + dataset + '_' + plottype + '_alg.pdf')