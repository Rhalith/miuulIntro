import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
"""
   survived  pclass     sex   age  sibsp  parch     fare embarked  class    who  adult_male deck  embark_town alive  alone
0         0       3    male  22.0      1      0   7.2500        S  Third    man        True  NaN  Southampton    no  False
1         1       1  female  38.0      1      0  71.2833        C  First  woman       False    C    Cherbourg   yes  False
2         1       3  female  26.0      0      0   7.9250        S  Third  woman       False  NaN  Southampton   yes   True
3         1       1  female  35.0      1      0  53.1000        S  First  woman       False    C  Southampton   yes  False
4         0       3    male  35.0      0      0   8.0500        S  Third    man        True  NaN  Southampton    no   True
"""
df.tail()
"""
     survived  pclass     sex   age  sibsp  parch   fare embarked   class    who  adult_male deck  embark_town alive  alone
886         0       2    male  27.0      0      0  13.00        S  Second    man        True  NaN  Southampton    no   True
887         1       1  female  19.0      0      0  30.00        S   First  woman       False    B  Southampton   yes   True
888         0       3  female   NaN      1      2  23.45        S   Third  woman       False  NaN  Southampton    no  False
889         1       1    male  26.0      0      0  30.00        C   First    man        True    C    Cherbourg   yes   True
890         0       3    male  32.0      0      0   7.75        Q   Third    man        True  NaN   Queenstown    no   True
"""
df.shape
"""
(891, 15)
"""
df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB
"""
df.columns
"""
Index(['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone'], dtype='object')
"""
df.index
"""
RangeIndex(start=0, stop=891, step=1)
"""
df.describe().T
"""
          count       mean        std   min      25%      50%   75%       max
survived  891.0   0.383838   0.486592  0.00   0.0000   0.0000   1.0    1.0000
pclass    891.0   2.308642   0.836071  1.00   2.0000   3.0000   3.0    3.0000
age       714.0  29.699118  14.526497  0.42  20.1250  28.0000  38.0   80.0000
sibsp     891.0   0.523008   1.102743  0.00   0.0000   0.0000   1.0    8.0000
parch     891.0   0.381594   0.806057  0.00   0.0000   0.0000   0.0    6.0000
fare      891.0  32.204208  49.693429  0.00   7.9104  14.4542  31.0  512.3292
"""
df.isnull().values.any()
"""
True
"""
df.isnull().sum()
"""
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64
"""

def CheckDataFrame(dataFrame, head=5):
    print("###### Shape ######")
    print(dataFrame.shape)
    print("###### Types ######")
    print(dataFrame.dtypes)
    print("###### Head ######")
    print(dataFrame.head(head))
    print("###### Tail ######")
    print(dataFrame.tail(head))
    print("###### NA ######")
    print(dataFrame.isnull().sum())
    print("###### Quantities ######")
    print(dataFrame.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

CheckDataFrame(df)
"""
###### Shape ######
(891, 15)
###### Types ######
survived          int64
pclass            int64
sex              object
age             float64
sibsp             int64
parch             int64
fare            float64
embarked         object
class          category
who              object
adult_male         bool
deck           category
embark_town      object
alive            object
alone              bool
dtype: object
###### Head ######
   survived  pclass     sex   age  sibsp  parch     fare embarked  class    who  adult_male deck  embark_town alive  alone
0         0       3    male  22.0      1      0   7.2500        S  Third    man        True  NaN  Southampton    no  False
1         1       1  female  38.0      1      0  71.2833        C  First  woman       False    C    Cherbourg   yes  False
2         1       3  female  26.0      0      0   7.9250        S  Third  woman       False  NaN  Southampton   yes   True
3         1       1  female  35.0      1      0  53.1000        S  First  woman       False    C  Southampton   yes  False
4         0       3    male  35.0      0      0   8.0500        S  Third    man        True  NaN  Southampton    no   True
###### Tail ######
     survived  pclass     sex   age  sibsp  parch   fare embarked   class    who  adult_male deck  embark_town alive  alone
886         0       2    male  27.0      0      0  13.00        S  Second    man        True  NaN  Southampton    no   True
887         1       1  female  19.0      0      0  30.00        S   First  woman       False    B  Southampton   yes   True
888         0       3  female   NaN      1      2  23.45        S   Third  woman       False  NaN  Southampton    no  False
889         1       1    male  26.0      0      0  30.00        C   First    man        True    C    Cherbourg   yes   True
890         0       3    male  32.0      0      0   7.75        Q   Third    man        True  NaN   Queenstown    no   True
###### NA ######
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64
###### Quantities ######
          count       mean        std   min    0%     5%      50%        95%        99%      100%       max
survived  891.0   0.383838   0.486592  0.00  0.00  0.000   0.0000    1.00000    1.00000    1.0000    1.0000
pclass    891.0   2.308642   0.836071  1.00  1.00  1.000   3.0000    3.00000    3.00000    3.0000    3.0000
age       714.0  29.699118  14.526497  0.42  0.42  4.000  28.0000   56.00000   65.87000   80.0000   80.0000
sibsp     891.0   0.523008   1.102743  0.00  0.00  0.000   0.0000    3.00000    5.00000    8.0000    8.0000
parch     891.0   0.381594   0.806057  0.00  0.00  0.000   0.0000    2.00000    4.00000    6.0000    6.0000
fare      891.0  32.204208  49.693429  0.00  0.00  7.225  14.4542  112.07915  249.00622  512.3292  512.3292
"""

df2 = sns.load_dataset("tips")

CheckDataFrame(df2)
"""
###### Shape ######
(244, 7)
###### Types ######
total_bill     float64
tip            float64
sex           category
smoker        category
day           category
time          category
size             int64
dtype: object
###### Head ######
   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4
###### Tail ######
     total_bill   tip     sex smoker   day    time  size
239       29.03  5.92    Male     No   Sat  Dinner     3
240       27.18  2.00  Female    Yes   Sat  Dinner     2
241       22.67  2.00    Male    Yes   Sat  Dinner     2
242       17.82  1.75    Male     No   Sat  Dinner     2
243       18.78  3.00  Female     No  Thur  Dinner     2
###### NA ######
total_bill    0
tip           0
sex           0
smoker        0
day           0
time          0
size          0
dtype: int64
###### Quantities ######
            count       mean       std   min    0%      5%     50%      95%      99%   100%    max
total_bill  244.0  19.785943  8.902412  3.07  3.07  9.5575  17.795  38.0610  48.2270  50.81  50.81
tip         244.0   2.998279  1.383638  1.00  1.00  1.4400   2.900   5.1955   7.2145  10.00  10.00
size        244.0   2.569672  0.951100  1.00  1.00  2.0000   2.000   4.0000   6.0000   6.00   6.00
"""

df["embarked"].value_counts()
"""
S    644
C    168
Q     77
Name: embarked, dtype: int64
"""
df["sex"].unique()
"""
array(['male', 'female'], dtype=object)
"""
df["class"].nunique()
"""
3
"""

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
cat_cols
"""
['sex',
 'embarked',
 'class',
 'who',
 'adult_male',
 'deck',
 'embark_town',
 'alive',
 'alone']
"""

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat
"""
['sex',
 'embarked',
 'class',
 'who',
 'adult_male',
 'deck',
 'embark_town',
 'alive',
 'alone']
"""

def CatSummary(dataFrame, colName):
    print(pd.DataFrame({colName: dataFrame[colName].value_counts(),
                        "Ratio": 100*dataFrame[colName].value_counts() / len(dataFrame)}))
CatSummary(df, "survived")
"""
   survived      Ratio
0       549  61.616162
1       342  38.383838
"""

for col in cat_cols:
    CatSummary(df, col)
"""
        sex      Ratio
male    577  64.758698
female  314  35.241302
   embarked      Ratio
S       644  72.278339
C       168  18.855219
Q        77   8.641975
        class      Ratio
Third     491  55.106622
First     216  24.242424
Second    184  20.650954
       who      Ratio
man    537  60.269360
woman  271  30.415264
child   83   9.315376
       adult_male     Ratio
True          537  60.26936
False         354  39.73064
   deck     Ratio
C    59  6.621773
B    47  5.274972
D    33  3.703704
E    32  3.591470
A    15  1.683502
F    13  1.459035
G     4  0.448934
             embark_town      Ratio
Southampton          644  72.278339
Cherbourg            168  18.855219
Queenstown            77   8.641975
     alive      Ratio
no     549  61.616162
yes    342  38.383838
       alone     Ratio
True     537  60.26936
False    354  39.73064
"""

def CatSummary(dataFrame, colName, plot = False):
    print(pd.DataFrame({colName: dataFrame[colName].value_counts(),
                        "Ratio": 100*dataFrame[colName].value_counts() / len(dataFrame)}))
    print("##############################")

    if plot:
        sns.countplot(x=dataFrame[colName], data=dataFrame)
        plt.show(block = True)

CatSummary(df, "sex", True)

for col in cat_cols:
    if df[col].dtypes != "bool":
        CatSummary(df, col, True)
    else:
        df[col] = df[col].astype(int)
        CatSummary(df, col, True)
"""
        sex      Ratio
male    577  64.758698
female  314  35.241302
##############################
   embarked      Ratio
S       644  72.278339
C       168  18.855219
Q        77   8.641975
##############################
        class      Ratio
Third     491  55.106622
First     216  24.242424
Second    184  20.650954
##############################
       who      Ratio
man    537  60.269360
woman  271  30.415264
child   83   9.315376
##############################
   adult_male     Ratio
1         537  60.26936
0         354  39.73064
##############################
   deck     Ratio
C    59  6.621773
B    47  5.274972
D    33  3.703704
E    32  3.591470
A    15  1.683502
F    13  1.459035
G     4  0.448934
##############################
             embark_town      Ratio
Southampton          644  72.278339
Cherbourg            168  18.855219
Queenstown            77   8.641975
##############################
     alive      Ratio
no     549  61.616162
yes    342  38.383838
##############################
   alone     Ratio
1    537  60.26936
0    354  39.73064
##############################
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

df[["age", "fare"]].describe()
"""
              age        fare
count  714.000000  891.000000
mean    29.699118   32.204208
std     14.526497   49.693429
min      0.420000    0.000000
25%     20.125000    7.910400
50%     28.000000   14.454200
75%     38.000000   31.000000
max     80.000000  512.329200
"""
df[["age", "fare"]].describe().T
"""
      count       mean        std   min      25%      50%   75%       max
age   714.0  29.699118  14.526497  0.42  20.1250  28.0000  38.0   80.0000
fare  891.0  32.204208  49.693429  0.00   7.9104  14.4542  31.0  512.3292
"""

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]

num_cols = [col for col in num_cols if col not in cat_cols]

num_cols
"""
['age', 'fare']
"""

def NumSummary(dataFrame, numericalCol):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataFrame[numericalCol].describe(quantiles).T)
for col in num_cols:
    NumSummary(df, col)
"""
count    714.000000
mean      29.699118
std       14.526497
min        0.420000
5%         4.000000
10%       14.000000
20%       19.000000
30%       22.000000
40%       25.000000
50%       28.000000
60%       31.800000
70%       36.000000
80%       41.000000
90%       50.000000
95%       56.000000
99%       65.870000
max       80.000000
Name: age, dtype: float64
count    891.000000
mean      32.204208
std       49.693429
min        0.000000
5%         7.225000
10%        7.550000
20%        7.854200
30%        8.050000
40%       10.500000
50%       14.454200
60%       21.679200
70%       27.000000
80%       39.687500
90%       77.958300
95%      112.079150
99%      249.006220
max      512.329200
Name: fare, dtype: float64
"""

def NumSummary(dataFrame, numericalCol, plot = False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataFrame[numericalCol].describe(quantiles).T)

    if plot:
        dataFrame[numericalCol].hist()
        plt.xlabel(numericalCol)
        plt.title(numericalCol)
        plt.show(block = True)

for col in num_cols:
    NumSummary(df, col, True)