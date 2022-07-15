import pandas as pd

s = pd.Series([10,77,12,4,5])

s.index
s.dtype
s.size
s.ndim
s.values

type(s.values)

s.head(3)
s.tail(3)


#df = pd.read_csv(path)

import seaborn as sns

df = sns.load_dataset("titanic")

df.head()

df.tail()

df.shape

df.info()

df.describe().T

df.isnull().values.any()

type(df.isnull().values)

df.isnull().sum() #truelar 1 falselar 0