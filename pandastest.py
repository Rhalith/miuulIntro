import pandas as pd

s = pd.Series([10,77,12,4,5])

s.index # RangeIndex(start=0, stop=5, step=1)
s.dtype # dtype('int64')
s.size # 5
s.ndim # 1
s.values # array([10, 77, 12,  4,  5], dtype=int64)

type(s.values) # numpy.ndarray

s.head(3)
#0    10
#1    77
#2    12
#dtype: int64
s.tail(3)
#2    12
#3     4
#4     5
#dtype: int64

#df = pd.read_csv(path)

import seaborn as sns

df = sns.load_dataset("titanic")

df.head()
"""   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0         0       3    male  22.0  ...   NaN  Southampton     no  False
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
4         0       3    male  35.0  ...   NaN  Southampton     no   True"""
df.tail()
"""     survived  pclass     sex   age  ...  deck  embark_town  alive  alone
886         0       2    male  27.0  ...   NaN  Southampton     no   True
887         1       1  female  19.0  ...     B  Southampton    yes   True
888         0       3  female   NaN  ...   NaN  Southampton     no  False
889         1       1    male  26.0  ...     C    Cherbourg    yes   True
890         0       3    male  32.0  ...   NaN   Queenstown     no   True"""

df.shape # (891, 15)

df.info()
"""<class 'pandas.core.frame.DataFrame'>
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

df.describe().T
"""          count       mean        std   min      25%      50%   75%       max
survived  891.0   0.383838   0.486592  0.00   0.0000   0.0000   1.0    1.0000
pclass    891.0   2.308642   0.836071  1.00   2.0000   3.0000   3.0    3.0000
age       714.0  29.699118  14.526497  0.42  20.1250  28.0000  38.0   80.0000
sibsp     891.0   0.523008   1.102743  0.00   0.0000   0.0000   1.0    8.0000
parch     891.0   0.381594   0.806057  0.00   0.0000   0.0000   0.0    6.0000
fare      891.0  32.204208  49.693429  0.00   7.9104  14.4542  31.0  512.3292"""

df.isnull().values.any() #True


type(df.isnull().values) #numpy.ndarray

df.isnull().sum() #truelar 1 falselar 0
"""survived         0
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
df["sex"].head()
"""0      male
1    female
2    female
3    female
4      male
Name: sex, dtype: object"""
df["sex"].value_counts()
"""
male      577
female    314
Name: sex, dtype: int64
"""

df.index
"""RangeIndex(start=0, stop=891, step=1)"""
df[0:13]
"""    survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0          0       3    male  22.0  ...   NaN  Southampton     no  False
1          1       1  female  38.0  ...     C    Cherbourg    yes  False
2          1       3  female  26.0  ...   NaN  Southampton    yes   True
3          1       1  female  35.0  ...     C  Southampton    yes  False
4          0       3    male  35.0  ...   NaN  Southampton     no   True
5          0       3    male   NaN  ...   NaN   Queenstown     no   True
6          0       1    male  54.0  ...     E  Southampton     no   True
7          0       3    male   2.0  ...   NaN  Southampton     no  False
8          1       3  female  27.0  ...   NaN  Southampton    yes  False
9          1       2  female  14.0  ...   NaN    Cherbourg    yes  False
10         1       3  female   4.0  ...     G  Southampton    yes  False
11         1       1  female  58.0  ...     C  Southampton    yes   True
12         0       3    male  20.0  ...   NaN  Southampton     no   True"""

df.drop(0,axis=0).head()
"""   survived  pclass     sex   age  ...  deck  embark_town  alive  alone
1         1       1  female  38.0  ...     C    Cherbourg    yes  False
2         1       3  female  26.0  ...   NaN  Southampton    yes   True
3         1       1  female  35.0  ...     C  Southampton    yes  False
4         0       3    male  35.0  ...   NaN  Southampton     no   True
5         0       3    male   NaN  ...   NaN   Queenstown     no   True"""

delete_indexes = [1,3,5,7]

df.drop(delete_indexes, axis=0).head(10)
"""    survived  pclass     sex   age  ...  deck  embark_town  alive  alone
0          0       3    male  22.0  ...   NaN  Southampton     no  False
2          1       3  female  26.0  ...   NaN  Southampton    yes   True
4          0       3    male  35.0  ...   NaN  Southampton     no   True
6          0       1    male  54.0  ...     E  Southampton     no   True
8          1       3  female  27.0  ...   NaN  Southampton    yes  False
9          1       2  female  14.0  ...   NaN    Cherbourg    yes  False
10         1       3  female   4.0  ...     G  Southampton    yes  False
11         1       1  female  58.0  ...     C  Southampton    yes   True
12         0       3    male  20.0  ...   NaN  Southampton     no   True
13         0       3    male  39.0  ...   NaN  Southampton     no  False"""

df["age"].head()
"""0    22.0
1    38.0
2    26.0
3    35.0
4    35.0
Name: age, dtype: float64"""

df.index = df["age"]
"""Float64Index([22.0, 38.0, 26.0, 35.0, 35.0,  nan, 54.0,  2.0, 27.0, 14.0,
              ...
              33.0, 22.0, 28.0, 25.0, 39.0, 27.0, 19.0,  nan, 26.0, 32.0],
             dtype='float64', name='age', length=891)"""

df.drop("age", axis=1).head() #kalıcı değil

df.drop("age", axis=1, inplace=True) #kalıcı
df.head()
"""      survived  pclass     sex  sibsp  ...  deck  embark_town alive  alone
age                                    ...                                
22.0         0       3    male      1  ...   NaN  Southampton    no  False
38.0         1       1  female      1  ...     C    Cherbourg   yes  False
26.0         1       3  female      0  ...   NaN  Southampton   yes   True
35.0         1       1  female      1  ...     C  Southampton   yes  False
35.0         0       3    male      0  ...   NaN  Southampton    no   True"""


df["age"] = df.index

df.head()

"""      survived  pclass     sex  sibsp  ...  embark_town  alive  alone   age
age                                    ...                                 
22.0         0       3    male      1  ...  Southampton     no  False  22.0
38.0         1       1  female      1  ...    Cherbourg    yes  False  38.0
26.0         1       3  female      0  ...  Southampton    yes   True  26.0
35.0         1       1  female      1  ...  Southampton    yes  False  35.0
35.0         0       3    male      0  ...  Southampton     no   True  35.0"""

df.reset_index().head() #before using it execute the df.drop("age", axis=1, inplace=True) otherwise you will get error.
"""    age  survived  pclass     sex  ...  deck  embark_town  alive  alone
0  22.0         0       3    male  ...   NaN  Southampton     no  False
1  38.0         1       1  female  ...     C    Cherbourg    yes  False
2  26.0         1       3  female  ...   NaN  Southampton    yes   True
3  35.0         1       1  female  ...     C  Southampton    yes  False
4  35.0         0       3    male  ...   NaN  Southampton     no   True"""

pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

"age" in df #True

type(df["age"].head()) # pandas.core.series.Series

df[["age", "alive"]].head()
"""    
    age alive
0  22.0    no
1  38.0   yes
2  26.0   yes
3  35.0   yes
4  35.0    no"""

df["age2"] = df["age"]**2
df.head()
"""
   survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
0         0       3    male  22.0      1      0   7.2500        S  Third   
1         1       1  female  38.0      1      0  71.2833        C  First   
2         1       3  female  26.0      0      0   7.9250        S  Third   
3         1       1  female  35.0      1      0  53.1000        S  First   
4         0       3    male  35.0      0      0   8.0500        S  Third   
     who  adult_male deck  embark_town alive  alone    age2  
0    man        True  NaN  Southampton    no  False   484.0  
1  woman       False    C    Cherbourg   yes  False  1444.0  
2  woman       False  NaN  Southampton   yes   True   676.0  
3  woman       False    C  Southampton   yes  False  1225.0  
4    man        True  NaN  Southampton    no   True  1225.0  
"""

df.drop("age2", axis=1).head()
"""
   survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
0         0       3    male  22.0      1      0   7.2500        S  Third   
1         1       1  female  38.0      1      0  71.2833        C  First   
2         1       3  female  26.0      0      0   7.9250        S  Third   
3         1       1  female  35.0      1      0  53.1000        S  First   
4         0       3    male  35.0      0      0   8.0500        S  Third   
     who  adult_male deck  embark_town alive  alone  
0    man        True  NaN  Southampton    no  False  
1  woman       False    C    Cherbourg   yes  False  
2  woman       False  NaN  Southampton   yes   True  
3  woman       False    C  Southampton   yes  False  
4    man        True  NaN  Southampton    no   True  
"""


df.iloc[0:3] #integer based selection
"""
   survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
0         0       3    male  22.0      1      0   7.2500        S  Third   
1         1       1  female  38.0      1      0  71.2833        C  First   
2         1       3  female  26.0      0      0   7.9250        S  Third   
     who  adult_male deck  embark_town alive  alone    age2  
0    man        True  NaN  Southampton    no  False   484.0  
1  woman       False    C    Cherbourg   yes  False  1444.0  
2  woman       False  NaN  Southampton   yes   True   676.0  
"""

df.loc[0:3] #label based selection
"""
   survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
0         0       3    male  22.0      1      0   7.2500        S  Third   
1         1       1  female  38.0      1      0  71.2833        C  First   
2         1       3  female  26.0      0      0   7.9250        S  Third   
3         1       1  female  35.0      1      0  53.1000        S  First   
     who  adult_male deck  embark_town alive  alone    age2  
0    man        True  NaN  Southampton    no  False   484.0  
1  woman       False    C    Cherbourg   yes  False  1444.0  
2  woman       False  NaN  Southampton   yes   True   676.0  
3  woman       False    C  Southampton   yes  False  1225.0  
"""

df.iloc[0:3, 0:3]
"""
   survived  pclass     sex
0         0       3    male
1         1       1  female
2         1       3  female
"""

df.loc[0:3, "age"]
"""
0    22.0
1    38.0
2    26.0
3    35.0
Name: age, dtype: float64
"""

df = sns.load_dataset("titanic")
df.head()

df[df["age"]>50].head()
"""
    survived  pclass     sex   age  sibsp  parch     fare embarked   class  \
6          0       1    male  54.0      0      0  51.8625        S   First   
11         1       1  female  58.0      0      0  26.5500        S   First   
15         1       2  female  55.0      0      0  16.0000        S  Second   
33         0       2    male  66.0      0      0  10.5000        S  Second   
54         0       1    male  65.0      0      1  61.9792        C   First   
      who  adult_male deck  embark_town alive  alone  
6     man        True    E  Southampton    no   True  
11  woman       False    C  Southampton   yes   True  
15  woman       False  NaN  Southampton   yes   True  
33    man        True  NaN  Southampton    no   True  
54    man        True    B    Cherbourg    no  False  
"""
df[df["age"]>50]["age"].count() #64

df.loc[df["age"]>50, "class"].head()
"""
6      First
11     First
15    Second
33    Second
54     First
Name: class, dtype: category
Categories (3, object): ['First', 'Second', 'Third']
"""

df.loc[df["age"]>50, ["age","class"]].head()
"""
     age   class
6   54.0   First
11  58.0   First
15  55.0  Second
33  66.0  Second
54  65.0   First
"""

df.loc[(df["age"]>50) & (df["sex"].__eq__("male")), ["age","class"]].head()
"""
     age   class
6   54.0   First
33  66.0  Second
54  65.0   First
94  59.0   Third
96  71.0   First
"""

df.loc[(df["age"]>50) & (df["sex"].__eq__("male") & (df["embark_town"].__eq__("Cherbourg"))), ["age","class", "embark_town"]].head()
"""
      age  class embark_town
54   65.0  First   Cherbourg
96   71.0  First   Cherbourg
155  51.0  First   Cherbourg
174  56.0  First   Cherbourg
487  58.0  First   Cherbourg
"""

df = sns.load_dataset("titanic")
df.head()

df["age"].mean() #29.69911764705882

df.groupby("sex")["age"].mean()
"""
sex
female    27.915709
male      30.726645
Name: age, dtype: float64
"""

df.groupby("sex").agg({"age": "mean"})
"""
              age
sex              
female  27.915709
male    30.726645
"""
df.groupby("sex").agg({"age": ["mean", "sum"]})
"""
              age          
             mean       sum
sex                        
female  27.915709   7286.00
male    30.726645  13919.17
"""

df.groupby(["sex", "embark_town"]).agg({"age": ["mean"], "survived": "mean"})
"""
                          age  survived
                         mean      mean
sex    embark_town                     
female Cherbourg    28.344262  0.876712
       Queenstown   24.291667  0.750000
       Southampton  27.771505  0.689655
male   Cherbourg    32.998841  0.305263
       Queenstown   30.937500  0.073171
       Southampton  30.291440  0.174603
"""

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"], "survived": "mean", "sex": "count"})
"""
                                 age  survived   sex
                                mean      mean count
sex    embark_town class                            
female Cherbourg   First   36.052632  0.976744    43
                   Second  19.142857  1.000000     7
                   Third   14.062500  0.652174    23
       Queenstown  First   33.000000  1.000000     1
                   Second  30.000000  1.000000     2
                   Third   22.850000  0.727273    33
       Southampton First   32.704545  0.958333    48
                   Second  29.719697  0.910448    67
                   Third   23.223684  0.375000    88
male   Cherbourg   First   40.111111  0.404762    42
                   Second  25.937500  0.200000    10
                   Third   25.016800  0.232558    43
       Queenstown  First   44.000000  0.000000     1
                   Second  57.000000  0.000000     1
                   Third   28.142857  0.076923    39
       Southampton First   41.897188  0.354430    79
                   Second  30.875889  0.154639    97
                   Third   26.574766  0.128302   265
"""

df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embark_town")
"""
embark_town  Cherbourg  Queenstown  Southampton
sex                                            
female        0.876712    0.750000     0.689655
male          0.305263    0.073171     0.174603
"""

df.pivot_table("survived", "sex", "embark_town", aggfunc="std")
"""
embark_town  Cherbourg  Queenstown  Southampton
sex                                            
female        0.331042    0.439155     0.463778
male          0.462962    0.263652     0.380058
"""

df.pivot_table("survived", "sex", ["embarked", "class"])
"""
embarked         C                      Q                          S  \
class        First Second     Third First Second     Third     First   
sex                                                                    
female    0.976744    1.0  0.652174   1.0    1.0  0.727273  0.958333   
male      0.404762    0.2  0.232558   0.0    0.0  0.076923  0.354430   
embarked                      
class       Second     Third  
sex                           
female    0.910448  0.375000  
male      0.154639  0.128302  
"""

df["new_age"] = pd.cut(df["age"], [0,10,18,25,40,90]) #if you know the data use cut, if you do not know and want to cut it quarter use qcut
df.head()
"""
   survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
0         0       3    male  22.0      1      0   7.2500        S  Third   
1         1       1  female  38.0      1      0  71.2833        C  First   
2         1       3  female  26.0      0      0   7.9250        S  Third   
3         1       1  female  35.0      1      0  53.1000        S  First   
4         0       3    male  35.0      0      0   8.0500        S  Third   
     who  adult_male deck  embark_town alive  alone   new_age  
0    man        True  NaN  Southampton    no  False  (18, 25]  
1  woman       False    C    Cherbourg   yes  False  (25, 40]  
2  woman       False  NaN  Southampton   yes   True  (25, 40]  
3  woman       False    C  Southampton   yes  False  (25, 40]  
4    man        True  NaN  Southampton    no   True  (25, 40]  
"""
df.pivot_table("survived", "sex", "new_age") # pivot_table(what you want, vertical column, horizontal column)
"""
new_age   (0, 10]  (10, 18]  (18, 25]  (25, 40]  (40, 90]
sex                                                      
female   0.612903  0.729730  0.759259  0.802198  0.770833
male     0.575758  0.131579  0.120370  0.220930  0.176471
"""

df.pivot_table("survived", "sex", ["new_age", "class"])
"""
new_age (0, 10]                   (10, 18]                   (18, 25]  \
class     First Second     Third     First Second     Third     First   
sex                                                                     
female      0.0    1.0  0.500000  1.000000    1.0  0.523810  0.941176   
male        1.0    1.0  0.363636  0.666667    0.0  0.103448  0.333333   
new_age                      (25, 40]                      (40, 90]            \
class      Second     Third     First    Second     Third     First    Second   
sex                                                                             
female   0.933333  0.500000  1.000000  0.906250  0.464286  0.961538  0.846154   
male     0.047619  0.115385  0.513514  0.071429  0.172043  0.280000  0.095238   
new_age            
class       Third  
sex                
female   0.111111  
male     0.064516  
"""

pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5

(df["age"]/10).head()
"""
0    2.2
1    3.8
2    2.6
3    3.5
4    3.5
Name: age, dtype: float64
"""
(df["age2"]/10).head()
"""
0    4.4
1    7.6
2    5.2
3    7.0
4    7.0
Name: age2, dtype: float64
"""
(df["age3"]/10).head()
"""
0    11.0
1    19.0
2    13.0
3    17.5
4    17.5
Name: age3, dtype: float64
"""

for col in df.columns:
    if "age" in col:
        print(col)
"""
age
age2
age3
"""

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10

df.head()

df[["age", "age2", "age3"]].apply(lambda x: x/10).head()
df.loc[:, df.columns.str.contains("age")].apply(lambda  x:x/10).head()
"""
    age  age2  age3
0  0.22  0.44  1.10
1  0.38  0.76  1.90
2  0.26  0.52  1.30
3  0.35  0.70  1.75
4  0.35  0.70  1.75
"""

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x-x.mean()) / x.std()).head()
"""
        age      age2      age3
0 -0.530005 -0.530005 -0.530005
1  0.571430  0.571430  0.571430
2 -0.254646 -0.254646 -0.254646
3  0.364911  0.364911  0.364911
4  0.364911  0.364911  0.364911
"""

def StandartScaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(StandartScaler).head()
"""
        age      age2      age3
0 -0.530005 -0.530005 -0.530005
1  0.571430  0.571430  0.571430
2 -0.254646 -0.254646 -0.254646
3  0.364911  0.364911  0.364911
4  0.364911  0.364911  0.364911
"""

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(StandartScaler)

df.head()
"""
   survived  pclass     sex       age  sibsp  parch     fare embarked  class    who  adult_male deck  embark_town alive  alone      age2      age3
0         0       3    male -1.346261      1      0   7.2500        S  Third    man        True  NaN  Southampton    no  False -1.346261 -1.346261
1         1       1  female  0.995063      1      0  71.2833        C  First  woman       False    C    Cherbourg   yes  False  0.995063  0.995063
2         1       3  female -0.760930      0      0   7.9250        S  Third  woman       False  NaN  Southampton   yes   True -0.760930 -0.760930
3         1       1  female  0.556064      1      0  53.1000        S  First  woman       False    C  Southampton   yes  False  0.556064  0.556064
4         0       3    male  0.556064      0      0   8.0500        S  Third    man        True  NaN  Southampton    no   True  0.556064  0.556064
"""

