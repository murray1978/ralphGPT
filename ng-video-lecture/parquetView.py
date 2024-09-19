

import pandas as pd

splits = {'train': 'data/train-00000-of-00001-7dbf637c625b2eca.parquet', 'validation': 'data/validation-00000-of-00001-d6b971d30d86821d.parquet', 'test': 'data/test-00000-of-00001-267a288de295ca03.parquet'}
df = pd.read_parquet("hf://datasets/deu05232/multiwoz_v23/" + splits["train"])

# Load the MultiWOZ dataset (replace with the correct path to your dataset)
# df = pd.read_parquet('multi_woz_v22/train.parquet')

# Display the first few rows to understand the structure
print(df.head())

# Check the columns in the dataset
print(df.columns)

# Explore a specific dialogue example
print(df.iloc[0])
