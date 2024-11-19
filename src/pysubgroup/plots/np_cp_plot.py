import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dir = '/home/alexpuff/runtimes/FullTest/'

iris = 'realisticTests/iris.csv'
spam = 'realisticTests/spam.csv'
darwin = 'realisticTests/darwin.csv'
cook = 'realisticTests/cook.csv'
delivery = 'realisticTests/delivery.csv'



def get_df(path):
    df = pd.read_csv(path)
    df['Rows'] = df['Rows'].astype(int)
    df['Depth'] = df['Depth'].astype(int)
    df['Data Loaded'] = df['Data Loaded'].astype(float)
    df['Search Space'] = df['Search Space'].astype(float)
    df['SG Discovery'] = df['SG Discovery'].astype(float)
    df['Total'] = df['Total'].astype(float)
    df['RAM'] = df['RAM'].astype(float)
    df['VRAM'] = df['VRAM'].astype(float)
    return df

def get_avgs(df):
    #result = pd.DataFrame(columns=['Rows', 'Data Loaded', 'Search Space', 'SG Discovery', 'Total'])
    result = []
    for d in np.sort(pd.unique(df['Depth'])):
    #for d in range(2,101):
        times = df[df['Depth']==d][['Data Loaded', 'Search Space', 'SG Discovery', 'Total', 'RAM', 'VRAM']]
        times = times.mean(axis=0)
        result.append(times)
    return pd.concat(result, ignore_index=True, axis=1)

def filter_df(df, device, quality, algorithm, gpu='-'):
    filtered = df[(df['Device']==device)&(df['Quality']==quality) & (df['Algorithm']==algorithm) & (df['GPU']==gpu)]
    return filtered

def plot_for_dataset(filepath):
    cook_df = get_df(filepath)
    plottype = 'linear'



#cpu=get_avgs(filter_df(cpu_df, dataset, 1, 'Apriori'))
#gpu_2080=get_avgs(filter_df(gpu_df, dataset, 1, 'Apriori', 'NVIDIA GeForce RTX 2080 Ti'))
#gpu_6000=get_avgs(filter_df(gpu_df, dataset, 1, 'Apriori', 'NVIDIA RTX A6000'))
#hor_2080=get_avgs(filter_df(hor_df, dataset, 1, 'Apriori', 'NVIDIA GeForce RTX 2080 Ti'))
#hor_6000=get_avgs(filter_df(hor_df, dataset, 1, 'Apriori', 'NVIDIA RTX A6000'))

    #lengths = [2**n for n in range(1,100)]
    #depths = list(range(1,100))
    #lengths = list(range(2,101))

    cpu = get_avgs(cook_df[(cook_df['Device']=='CPU')])
    
    gpu_a6000 = get_avgs(filter_df(cook_df,'GPU',  1, 'Apriori', 'NVIDIA RTX A6000'))
    #gpu_2080ti = get_avgs(filter_df(cook_df,'GPU',  1, 'Apriori', 'NVIDIA GeForce RTX 2080 Ti'))
    hor_a6000 = get_avgs(filter_df(cook_df,'Horizontal',  1, 'Apriori', 'NVIDIA RTX A6000'))
    #hor_2080ti = get_avgs(filter_df(cook_df,'Horizontal',  1, 'Apriori', 'NVIDIA GeForce RTX 2080 Ti'))
    
#cpu_times = cpu.loc['Search Space'] + cpu.loc['SG Discovery']
#gpu_2080_times = gpu_2080.loc['Search Space'] + gpu_2080.loc['SG Discovery']
#gpu_6000_times = gpu_6000.loc['Search Space'] + gpu_6000.loc['SG Discovery']
#hor_2080_times = hor_2080.loc['Search Space'] + hor_2080.loc['SG Discovery']
#hor_6000_times = hor_6000.loc['Search Space'] + hor_6000.loc['SG Discovery']
    cpu_times = cpu.loc['RAM'] + cpu.loc['VRAM']# + cpu.loc['SG Discovery']
    gpu_a6000_times = gpu_a6000.loc['RAM'] + gpu_a6000.loc['VRAM']# + gpu_a6000.loc['SG Discovery']
    #gpu_2080ti_times = gpu_2080ti.loc['SG Discovery'] + gpu_2080ti.loc['Search Space']
    hor_a6000_times = hor_a6000.loc['RAM'] + hor_a6000.loc['VRAM']# + hor_a6000.loc['SG Discovery']
    #hor_2080ti_times = hor_2080ti.loc['SG Discovery'] + hor_2080ti.loc['Search Space']
    
    lengths = np.sort(pd.unique(cook_df['Rows']))
    lengths = list(range(1,15))


    if plottype == 'loglog':
        plot = plt.loglog
    if plottype == 'semilog':
        plot = plt.semilogx
    if plottype == 'linear':
         plot = plt.plot

    plt.figure(figsize=(11, 5))
    plot(lengths[:len(cpu_times)], cpu_times, label='CPU', color='blue')
    #plot(lengths[:len(gpu_2080ti_times)], gpu_2080ti_times, label='2080ti, vertical', color='green',linestyle='--')
    plot(lengths[:len(gpu_a6000_times)], gpu_a6000_times, label='A6000, vertical', color='green')
    #plot(lengths[:len(hor_2080ti_times)], hor_2080ti_times, label='2080ti, full', color='green')
    plot(lengths[:len(gpu_a6000_times)], hor_a6000_times, label='A6000, full', color='orange')

    #plt.title('Time For Subgroup Discovery Iris Dataset, Different Depths')
    plt.xlabel('Maximum Exploration Depth')
    plt.ylabel('Memory usage, in MB')


    #plt.title('Log Plot of CPU and GPU Runtimes for count_nonzero')
    plt.legend()
    plt.grid(True, which="major", ls="--")

    # Save the plot as PDF
    plt.savefig(dir + 'plots/' + 'memdepth' + '_' + plottype + '_alg.pdf')
    
    
def plot_multi(paths):
    #titles = ['Iris', 'Spam', 'Darwin']
    lengths = [2**n for n in range(1,100)]
    #fig, axs = plt.subplots(figsize = (11,11), nrows = len(paths), sharex= True)
    #fig.suptitle('Time For Subgorup Discovery On Synthtically Expanded Datasets')
    titles = ['Iris', 'Spam', 'Darwin']
    #lengths = range(1,100)
    fig, axs = plt.subplots(figsize = (11,11), nrows = len(paths), sharex= True)
    
    for i in range(len(paths)):
        df = get_df(dir + paths[i])
        cpu = get_avgs(df[(df['Device']=='CPU')])
        gpu_a6000 = get_avgs(filter_df(df,'GPU',  1, 'Apriori', 'NVIDIA RTX A6000'))
        gpu_2080ti = get_avgs(filter_df(df,'GPU',  1, 'Apriori', 'NVIDIA GeForce RTX 2080 Ti'))
        hor_a6000 = get_avgs(filter_df(df,'Horizontal',  1, 'Apriori', 'NVIDIA RTX A6000'))
        hor_2080ti = get_avgs(filter_df(df,'Horizontal',  1, 'Apriori', 'NVIDIA GeForce RTX 2080 Ti'))
        cpu_times = cpu.loc['Search Space'] + cpu.loc['SG Discovery']
        gpu_a6000_times = gpu_a6000.loc['Search Space'] + gpu_a6000.loc['SG Discovery']
        gpu_2080ti_times = gpu_2080ti.loc['Search Space'] + gpu_2080ti.loc['SG Discovery']
        hor_a6000_times = hor_a6000.loc['Search Space'] + hor_a6000.loc['SG Discovery']
        hor_2080ti_times = hor_2080ti.loc['Search Space'] + hor_2080ti.loc['SG Discovery']
        axs[i].plot(lengths[:len(cpu_times)], cpu_times, label='CPU', color='blue')
        axs[i].semilogx(lengths[:len(gpu_2080ti_times)], gpu_2080ti_times, label='2080ti, vertical', color='green',linestyle='--')
        axs[i].plot(lengths[:len(gpu_a6000_times)], gpu_a6000_times, label='A6000, vertical', color='green')
        axs[i].semilogx(lengths[:len(hor_2080ti_times)], hor_2080ti_times, label='2080ti, full', color='green')
        axs[i].plot(lengths[:len(hor_a6000_times)], hor_a6000_times, label='A6000, full', color='orange')
        axs[i].grid(True, which="major", ls="--")
        axs[i].set_title(titles[i])
        axs[i].set_ylabel('Time (Seconds)')
        axs[i].set_xlabel('Number of Rows')
    
    axs[0].legend(loc='upper left')
    #fig.tight_layout()
    
    
    plt.savefig(dir + 'plots/'  + 'synth_multi.pdf')
    
def plot_multi_qualities(paths):
    #titles = ['Iris', 'Spam', 'Darwin']
    #lengths = [2**n for n in range(1,100)]
    #fig, axs = plt.subplots(figsize = (11,11), nrows = len(paths), sharex= True)
    #fig.suptitle('Time For Subgorup Discovery On Synthtically Expanded Datasets')
    titles = ['CPU', 'Vertical', 'Horizontal']
    lengths = [2**n for n in range(100)]
    fig, axs = plt.subplots(figsize = (11,11), nrows = len(paths), sharex= True)
    
    for i in range(len(paths)):
        df = get_df(dir + paths[i])
        t0 = get_avgs(df[df['Quality'] == 0])
        t01 = get_avgs(df[df['Quality'] == 0.1])
        t05 = get_avgs(df[df['Quality'] == 0.5])
        t1 = get_avgs(df[df['Quality'] == 1])
        times0 = t0.loc['Search Space'] + t0.loc['SG Discovery']
        times01 = t01.loc['Search Space'] + t01.loc['SG Discovery']
        times05 = t05.loc['Search Space'] + t05.loc['SG Discovery']
        times1 = t1.loc['Search Space'] + t1.loc['SG Discovery']
        axs[i].semilogx(lengths[:len(times0)], times0, label=r'$\alpha = 0.0$', color='blue')
        axs[i].semilogx(lengths[:len(times01)], times01, label=r'$\alpha = 0.1$', color='green')
        axs[i].semilogx(lengths[:len(times05)], times05, label=r'$\alpha = 0.5$', color='orange')
        axs[i].semilogx(lengths[:len(times1)], times1, label=r'$\alpha = 1.0$', color='red')
        axs[i].grid(True, which="major", ls="--")
        axs[i].set_title(titles[i])
        axs[i].set_ylabel('Time (Seconds)')
    plt.xlabel('Number of Rows')
    
    axs[0].legend(loc='upper left')
    fig.tight_layout()
    
    
    plt.savefig(dir + 'plots/'  + 'qualities.pdf')

plot_for_dataset(dir + '/memory/memdepth.csv')
