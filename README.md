<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/pysubgroup)
-->

<!--![Build status](https://github.com/flemmerich/pysubgroup/actions/workflows/ci.yaml/badge.svg)
[![ReadTheDocs](https://readthedocs.org/projects/pysubgroup/badge/?version=latest)](https://pysubgroup.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/flemmerich/pysubgroup/main.svg)](https://coveralls.io/r/flemmerich/pysubgroup)
[![PyPI-Server](https://img.shields.io/pypi/v/pysubgroup.svg)](https://pypi.org/project/pysubgroup/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/pysubgroup.svg)](https://anaconda.org/conda-forge/pysubgroup)
[![Monthly Downloads](https://pepy.tech/badge/pysubgroup/month)](https://pepy.tech/project/pysubgroup)-->

# Accelerating pysubgroup with GPUs

This fork of [**pysubgroup**](https://github.com/flemmerich/pysubgroup) greatly speeds up run times by providing additional, GPU accelerated implementations of subgroup discovery. This update was part of my Master's Thesis at the University of Passau, supervised by professor Lemmerich.

Subgroup discovery lends itself well to GPU acceleration: It consists mainly of independent, naturally vectorized computations, for example counting the size of a subgroup (equivalent to summing up a boolean vector) and computing the intersection of two subgroups (logical and between two vectors). The speed-up over the sequential CPU implementation varies based on the dataset and chosen parameters, but consistently stays well above 10x. In the best cases, the GPU is more than 100x faster: For example, Apriori on 4 million rows of the [cooking time](https://github.com/yandex-research/tabred) dataset with 195 features takes only 12s on the GPU, while the CPU needs 11 minutes.
## Main Updates

I implemented GPU-accelerated subgroup discovery in two steps. The original code mainly uses NumPy for the computations mentioned above, so simply adding their [**CuPy**](https://cupy.dev/) equivalents to the original code is almost enough to get some baseline acceleration going. The data is loaded directly to GPU using [**cuDF**](https://docs.rapids.ai/api/cudf/stable/), a pandas-like package for manipulating data frames directly in VRAM. Using this updated code is as simple as loading the data with cuDF, instead of pandas, see the usage section below.

While this surprisingly simple update already greatly speeds up subgroup discovery in some cases, especially tall datasets with few columns and many rows, an obvious bottleneck was quickly identified: Individual subgroups are only ever evaluated one at a time. The original code explores the search space iteratively or recursively, depending on the algorithm. To this end, I implemented a more extensive update, which now evaluates large amounts of subgroups in parallel, only limited by VRAM. Unlike the simple update this does not reuse original pysubgroup code, full parallelization is contained in the `horizontal.py` and `gpu_algorithms.py` files.



## Additional Requirements

Of course, to benefit from GPU parallelization a modern, graphics card is required. While other manufacturers provide APIs similar to CUDA, they generally still lack its wide-spread support. Like already mentioned, I chose to implement most changes using CuPy and cuDF, both of which only work on **NVIDIA GPUs with [compute capability](https://developer.nvidia.com/cuda-gpus) 7.0 or higher**. Make sure a proper CUDA environment is set up, with updated GPU drivers and the [**CUDA Toolkit**](https://developer.nvidia.com/cuda-toolkit) installed. Sadly, cuDF only works on **Linux** machines, but Windows users can use a **WSL2** instance as a lightweight virtual environment.


## Installation

While this update still consists of pure Python code, the installation is not as straight-forward as before. After cloning this repository, first install **cuDF**, see their [installation instructions](https://docs.rapids.ai/install/), I recommend using conda. Choose your CUDA and Python versions, and select "Choose specific packages" to only install cuDF. This will create a venv, which you can use to install pysubgroup with a simple `pip install .`.

Note: Some WSL2 users get a `CUDA_ERROR_NO_DEVICE (100)` when trying to run the script. To fix this, you need to update your .bashrc file. Open it with nano:
```
sudo nano .bashrc
```
Insert these lines:
```
export LD_LIBRARY_PATH="/usr/lib/wsl/lib/"  
export NUMBA_CUDA_DRIVER="/usr/lib/wsl/lib/libcuda.so.1"
```
And reload the .bashrc:
```
source .bashrc
```
This solution was found on [stackoverflow](https://stackoverflow.com/questions/77380210/rapids-cannot-import-cudf-error-at-driver-init-call-to-cuinit-results-in-cuda).

## Usage
Here is how you can activate GPU acceleration while still using largely the same code as before:
```python
import pysubgroup as ps
import cudf

# Load a dataset
data = cudf.read_csv("path/to/data.csv")

# Specify target
target = ps.BinaryTarget ('Target_column', Target_value)

# Initialize search space
searchspace = ps.create_selectors(data, ignore=['Ignored_column'])

# Bundle everything in a task, here with weighted relative accuracy as quality function
task = ps.SubgroupDiscoveryTask (
    data,
    target,
    searchspace,
    result_set_size=5,
    depth=2,
    qf=ps.WRAccQF()
    )

# Execute task with one of two algorithms: DFS uses less memory but is a lot slower than Apriori
result = ps.DFS().execute(task)
result = ps.Apriori().execute(task)
```
Note that only DFS and Apriori support GPU acceleration, and only nominal or interval targets can be specified. Additionally, take extra care of the ignored columns: The target column should always be removed, and columns with missing values are not yet supported.

And for the much faster variant:
```python
import pysubgroup as ps
import cudf

# Load a dataset
data = cudf.read_csv("path/to/data.csv")

# Initialize search space
sp = ps.GpuSearchSpace(
    data, 'Target_column', 
    target_low = Lower_bound,
    target_high = Upper_bound,
    ignore=['Ignored_column']
    )

# Bundle into task
task = ps.GpuTask(
    sp,
    qf = 1,
    result_set_size = 5,
    depth = 2
    )

# Initialize algorithm
alg = ps.GpuBfs(task, apriori=True)

# Execute
result = alg.execute()
```

Targets are now always intervals, but nominal targets can be specified by leaving target_high blank. Quality functions are specified by a simple coefficient $a \in [0,1]$, with $1$ corresponding to weighted relative accuracy, and $0$ to a simple lift measure. I recommend using breadth-first search with Apriori enabled for the best results, although depth-first search is also supported.


 
## Limitations

Sadly the GPU libraries used here do generally not support **missing values**, datasets with them will lead to errors. Data types are also restricted: Only numerics support interval type targets and selectors, all other data types only support nominal targets and selectors. This is why, for example, datetime columns should either be converted into a numeric format (to allow intervals), or skipped entirely.

Like already mentioned above, this is not a feature-complete update for pysubgroup. Depth-first search and Apriori, as the two most common algorithms, are the only ones accelerated here. Additionally, numeric targets are not supported.

And finally, runtimes are only faster for certain dataset sizes, with the speed-up factor mainly increasing with row counts. To take full advantage of GPU acceleration, a decent amount of VRAM is  required.

<!--## Subgroup Discovery

Subgroup Discovery is a well established data mining technique that allows you to identify patterns in your data.
More precisely, the goal of subgroup discovery is to identify descriptions of data subsets that show an interesting distribution with respect to a pre-specified target concept.
For example, given a dataset of patients in a hospital, we could be interested in subgroups of patients, for which a certain treatment X was successful.
One example result could then be stated as:

_"While in general the operation is successful in only 60% of the cases", for the subgroup
of female patients under 50 that also have been treated with drug d, the success rate was 82%."_

Here, a variable _operation success_ is the target concept, the identified subgroup has the interpretable description _female=True AND age<50 AND drug_D = True_. We call these single conditions (such as _female=True_) selection expressions or short _selectors_.
The interesting behavior for this subgroup is that the distribution of the target concept differs significantly from the distribution in the overall general dataset.
A discovered subgroup could also be seen as a rule:
```
female=True AND age<50 AND drug_D = True ==> Operation_outcome=SUCCESS
```
Computationally, subgroup discovery is challenging since a large number of such conjunctive subgroup descriptions have to be considered. Of course, finding computable criteria, which subgroups are likely interesting to a user is also an eternal struggle.
Therefore, a lot of literature has been devoted to the topic of subgroup discovery (including some of my own work). Recent overviews on the topic are for example:

* Herrera, Franciso, et al. ["An overview on subgroup discovery: foundations and applications."](https://scholar.google.de/scholar?q=Herrera%2C+Franciso%2C+et+al.+%E2%80%9CAn+overview+on+subgroup+discovery%3A+foundations+and+applications.%E2%80%9D+Knowledge+and+information+systems+29.3+(2011)%3A+495-525.) Knowledge and information systems 29.3 (2011): 495-525.
* Atzmueller, Martin. ["Subgroup discovery."](https://scholar.google.de/scholar?q=Atzmueller%2C+Martin.+%E2%80%9CSubgroup+discovery.%E2%80%9D+Wiley+Interdisciplinary+Reviews%3A+Data+Mining+and+Knowledge+Discovery+5.1+(2015)%3A+35-49.) Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 5.1 (2015): 35-49.
* And of course, my point of view on the topic is [summarized in my dissertation](https://opus.bibliothek.uni-wuerzburg.de/files/9781/Dissertation-Lemmerich.pdf):

### Prerequisites and Installation
pysubgroup is built to fit in the standard Python data analysis environment from the scipy-stack.
Thus, it can be used just having pandas (including its dependencies numpy, scipy, and matplotlib) installed. Visualizations are carried out with the matplotlib library.

pysubgroup consists of pure Python code. Thus, you can simply download the code from the repository and copy it in your `site-packages` directory.
pysubgroup is also on PyPI and should be installable using:
`pip install pysubgroup`

**Note**: Some users complained about the **pip installation not working**.
If, after the installation, it still doesn't find the package, then do the following steps:
 1. Find where the directory `site-packages` is.
 2. Copy the folder `pysubgroup`, which contains the source code, into the `site-packages` directory. (WARNING: This is not the main repository folder. The `pysubgroup` folder is inside the main repository folder, at the same level as `doc`)
 3. Now you can import the module with `import pysubgroup`.

## How to use:
A simple use case (here using the well known _titanic_ data) can be created in just a few lines of code:

```python
import pysubgroup as ps

# Load the example dataset
from pysubgroup.datasets import get_titanic_data
data = get_titanic_data()

target = ps.BinaryTarget ('Survived', True)
searchspace = ps.create_selectors(data, ignore=['Survived'])
task = ps.SubgroupDiscoveryTask (
    data,
    target,
    searchspace,
    result_set_size=5,
    depth=2,
    qf=ps.WRAccQF())
result = ps.DFS().execute(task)
```
The first line imports _pysubgroup_ package.
The following lines load an example dataset (the popular titanic dataset).

Therafter, we define a target, i.e., the property we are mainly interested in (_'survived'}.
Then, we define the searchspace as a list of basic selectors. Descriptions are built from this searchspace. We can create this list manually, or use an utility function.
Next, we create a SubgroupDiscoveryTask object that encapsulates what we want to find in our search.
In particular, that comprises the target, the search space, the depth of the search (maximum numbers of selectors combined in a subgroup description), and the interestingness measure for candidate scoring (here, the Weighted Relative Accuracy measure).

The last line executes the defined task by performing a search with an algorithm---in this case depth first search. The result of this algorithm execution is stored in a SubgroupDiscoveryResults object.

To just print the result, we could for example do:

```python
print(result.to_dataframe())
```

to get:

<table border="1" class="dataframe">
<thead>    <tr style="text-align: right;">      <th></th>      <th>quality</th>      <th>description</th>    </tr>  </thead>
<tbody>
    <tr>      <th>0</th>      <td>0.132150</td>      <td>Sex==female</td>    </tr>
    <tr>      <th>1</th>      <td>0.101331</td>      <td>Parch==0 AND Sex==female</td>    </tr>
    <tr>      <th>2</th>      <td>0.079142</td>      <td>Sex==female AND SibSp: [0:1[</td>    </tr>
    <tr>      <th>3</th>      <td>0.077663</td>      <td>Cabin.isnull() AND Sex==female</td>    </tr>
    <tr>      <th>4</th>      <td>0.071746</td>      <td>Embarked==S AND Sex==female</td>    </tr>
</tbody></table>


## Key classes
Here is an outline on the most important classes:
* Selector: A Selector represents an atomic condition over the data, e.g., _age < 50_. There several subtypes of Selectors, i.e., NominalSelector (color==BLUE), NumericSelector (age < 50) and NegatedSelector (a wrapper such as not selector1)
* SubgroupDiscoveryTask: As mentioned before, encapsulates the specification of how an algorithm should search for interesting subgroups
* SubgroupDicoveryResult: These are the main outcome of a subgroup disovery run. You can obtain a list of subgroups using the `to_subgroups()` or to a dataframe using `to_dataframe()`
* Conjunction: A conjunction is the most widely used SubgroupDescription, and indicates which data instances are covered by the subgroup. It can be seen as the left hand side of a rule.


## License
We are happy about anyone using this software. Thus, this work is put under an Apache license. However, if this constitutes
any hindrance to your application, please feel free to contact us, we am sure that we can work something out.

    Copyright 2016-2019 Florian Lemmerich

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


## Warning
* GP-growth is in an experimental stage.

## Cite
If you are using pysubgroup for your research, please consider citing our demo paper:

    Lemmerich, F., & Becker, M. (2018, September). pysubgroup: Easy-to-use subgroup discovery in python. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECMLPKDD). pp. 658-662.

bibtex:

    @inproceedings{lemmerich2018pysubgroup,
      title={pysubgroup: Easy-to-use subgroup discovery in python},
      author={Lemmerich, Florian and Becker, Martin},
      booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
      pages={658--662},
      year={2018}
    }


## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.-->
