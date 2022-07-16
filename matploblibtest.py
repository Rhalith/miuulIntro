#Kategorik değişken: Sütun grafik countplot, bar
#Sayısal değişken: hist, boxplot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

#df["sex"].value_counts().plot(kind="bar")
#plt.show()

#plt.hist(df["age"])
#plt.show()

#plt.boxplot(df["fare"])
#plt.show()

import numpy as np

x = np.array([1,8])
y = np.array([0,150])

#plt.plot(x,y)
#plt.plot(x,y, "o")
#plt.show()

y = np.array([13,28,11,100])

#plt.plot(y, marker="*")
#plt.show()

plt.plot(y, linestyle = "dashdot", color = "r", marker = "*")
plt.plot(x)

#title
plt.title("Main title")

plt.xlabel("X")
plt.ylabel("Y")
plt.show()

x = np.array([1,8])
y = np.array([0,150])

plt.subplot(1,3,1)
plt.title("1")
plt.plot(x,y)

plt.subplot(1,3,2)
plt.title("2")
plt.plot(x,y)

plt.subplot(1,3,3)
plt.title("3")
plt.plot(x,y)

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()
"""
   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4
"""

df["sex"].value_counts()
"""
Male      157
Female     87
Name: sex, dtype: int64
"""
sns.countplot(x=df["sex"], data=df)
#same
df["sex"].value_counts().plot(kind="bar")

sns.boxplot(x=df["total_bill"])

df["total_bill"].hist()
plt.show()

plt