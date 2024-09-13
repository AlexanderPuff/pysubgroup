import sdv.datasets.demo as sdd
from sdv.single_table import GaussianCopulaSynthesizer
import pandas as pd

def load_titanic():
    from pysubgroup.datasets import get_titanic_data
    data = get_titanic_data()
    from sdv.metadata import SingleTableMetadata as stm
    metadata = stm()
    metadata.detect_from_dataframe(data)
    return data, metadata

def make_sampler(data, metadata):
    synth=GaussianCopulaSynthesizer(metadata)
    synth.fit(data)
    return synth


synth=GaussianCopulaSynthesizer.load(filepath='pysubgroup/src/pysubgroup/data/synthetic/titanic.pkl')
synth_data=synth.sample(num_rows=1000000)
synth_data.to_csv('pysubgroup/src/pysubgroup/data/synthetic/synth_titanic.csv', index=False)


