import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
def grab_col_names(dataFrame, cat_th=10, car_th=20):
    """
    Gives names of the categorical, numeric and categorical but cardinal variables in the dataframe.
    :param dataFrame: DataFrame
    :param cat_th: int, float
        max value for numeric but categorical variables
    :param car_th: int, float
        max value for categorical but cardinal variables
    :return:
        cat_cols: list
            categorical variables list
        num_cols: list
            numerical variables list
        cat_bur_car: list
            categorical but cardinal variables list
    :notes:
        cat_cols + num_cols + cat_but_car = total variable count
        num_but_cat is in cat_cols.
    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int", "float"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataFrame.shape[0]}")
    print(f"Variables: {dataFrame.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.groupby("sex")["survived"].mean()
"""
sex
female    0.742038
male      0.188908
Name: survived, dtype: float64
"""

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"Target_Mean": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

target_summary_with_cat(df, "survived", "sex")
"""
        Target_Mean
sex                
female     0.742038
male       0.188908
"""

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)
"""
        Target_Mean
sex                
female     0.742038
male       0.188908
          Target_Mean
embarked             
C            0.553571
Q            0.389610
S            0.336957
        Target_Mean
class              
First      0.629630
Second     0.472826
Third      0.242363
       Target_Mean
who               
child     0.590361
man       0.163873
woman     0.756458
      Target_Mean
deck             
A        0.466667
B        0.744681
C        0.593220
D        0.757576
E        0.750000
F        0.615385
G        0.500000
             Target_Mean
embark_town             
Cherbourg       0.553571
Queenstown      0.389610
Southampton     0.336957
       Target_Mean
alive             
no             0.0
yes            1.0
            Target_Mean
adult_male             
0              0.717514
1              0.163873
       Target_Mean
alone             
0         0.505650
1         0.303538
"""

df.groupby("survived")["age"].mean()
"""
survived
0    30.626179
1    28.343690
Name: age, dtype: float64
"""
df.groupby("survived").agg({"age":"mean"})
"""
                age
survived           
0         30.626179
1         28.343690
"""

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col : "mean"}), end="\n\n\n")
target_summary_with_num(df, "survived", "age")
"""
                age
survived           
0         30.626179
1         28.343690
"""

for col in num_cols:
    target_summary_with_num(df, "survived", col)