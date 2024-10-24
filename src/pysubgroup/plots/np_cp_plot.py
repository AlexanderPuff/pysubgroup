import matplotlib.pyplot as plt
import pandas as pd

dir = '/home/alexpuff/runtimes/FullTest/'

cpu = 'cpu.csv'
gpu = 'gpu.csv'
hor = 'horizontal.csv'



def get_df(path):
    df = pd.read_csv(path)
    df = df[df['Rows'] != 'Rows']
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


cpu_iris=get_avgs(cpu_df[cpu_df['Dataset']=='Iris'])
gpu_iris=get_avgs(gpu_df[gpu_df['Dataset']=='Iris'])
hor_iris=get_avgs(hor_df[hor_df['Dataset']=='Iris'])
cpu_spam=get_avgs(cpu_df[cpu_df['Dataset']=='Spam'])
gpu_spam=get_avgs(gpu_df[gpu_df['Dataset']=='Spam'])
hor_spam=get_avgs(hor_df[hor_df['Dataset']=='Spam'])

cpu_darwin=get_avgs(cpu_df[cpu_df['Dataset']=='Darwin'])
gpu_darwin=get_avgs(gpu_df[gpu_df['Dataset']=='Darwin'])
hor_darwin=get_avgs(hor_df[hor_df['Dataset']=='Darwin'])

lengths = pd.unique(cpu_df['Rows'])

cpu_times = cpu_iris.loc['Total']
gpu_times = gpu_iris.loc['Total']
hor_times = hor_iris.loc['Total']



plt.figure(figsize=(10, 6))
plt.semilogx(lengths[:len(cpu_times)], cpu_times, label='CPU', marker='o')
plt.semilogx(lengths[:len(gpu_times)], gpu_times, label='GPU', marker='o')
plt.semilogx(lengths[:len(hor_times)], hor_times, label='Full', marker='o')


plt.xlabel('Length')
plt.ylabel('Time (s)')
#plt.title('Log Plot of CPU and GPU Runtimes for count_nonzero')
plt.legend()
plt.grid(True, which="major", ls="--")

# Save the plot as PDF
plt.savefig(dir + 'plots/iris_total_semilog.pdf')