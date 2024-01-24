import numpy as np
import pandas as pd
from helpers.helpers import *
from helpers.eda import *
from helpers.data_prep import *

"""
Survived: hayatta kalma(0 = Hayır, 1 =Evet)
Pclass: bilet sınıfı(1 = 1., 2 = 2., 3 = 3.)
Sex: cinsiyet
Sibsp: Titanik’teki kardeş/eş sayısı
Parch: Titanik’teki ebeveynlerin/çocukların sayısı
Ticket: bilet numarası
Fare: ücret
Cabin: kabin numarası
Embarked: biniş limanı
"""

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def load():
    data = pd.read_csv("Datasets/titanic.csv")
    return data


df = load()
df.head()
check_df(df)

df.columns = [col.upper() for col in df.columns]

# Dealing with Outliers
num_cols = [col for col in df.columns if len(df[col].unique()) > 20
                and df[col].dtypes != 'O'
                and col not in "PassengerId"]

for col in num_cols:
    replace_with_thresholds(df, col)

# Dealing with null values.
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

# Feature Engineering
df["NEW_CABIN_BOOL"] = df["CABIN"].isnull().astype('int')
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
df["NEWfRD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

# Dropping useless columns.
df.drop(["TICKET", "NAME", "CABIN"], inplace=True, axis=1)

# Label Encoding
binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']

for col in binary_cols:
    df = label_encoder(df, col)

df = rare_encoder(df, 0.01)

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
df = one_hot_encoder(df, ohe_cols)

# Check DataFrame.
check_df(df)

# To pickle, csv.
df.to_pickle("Datasets/prepared_titanic_df.pkl")

df = pd.read_pickle("Datasets/prepared_titanic_df.pkl")

df.to_csv("prepared_titanic_df.csv")

df.shape