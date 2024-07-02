from sdv.metadata import SingleTableMetadata

d = SingleTableMetadata()
d.detect_from_csv(filepath='/home/alexpuff/datasets/titanic.csv')
print(d.to_dict())