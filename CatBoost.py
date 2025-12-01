import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from catboost import CatBoostClassifier, Pool



df = pd.read_csv("train_data.csv")
df = df.drop(['case_id', 'patientid'], axis=1)

LE = LabelEncoder() # To turn categorical values into numerical ones (Mainly used for Stay Feature)
df["Stay"] = LE.fit_transform(df["Stay"])

df['Admission_Deposit'] = pd.to_numeric(df['Admission_Deposit'], errors='coerce')

X = df.drop("Stay", axis = 1)
y = df["Stay"]

print("COLUMNS PASSED TO CATBOOST:")
print(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train.shape)
print(X_train.select_dtypes(include=['object', 'category']).columns)
print(X_train.nunique().sort_values(ascending=False).head(10))

cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
cat_idx = [X_train.columns.get_loc(col) for col in cat_cols]

print("TRAINING DF SHAPE:", df.shape)
print(df.head())
print(df.dtypes)

'''
EXCESS TEST PRINT STATEMENTS
print("cat_idx =", cat_idx)
print("Number of categorical columns:", len(cat_idx))
print("TRAIN POOL SHAPE:", train_pool.num_row(), "rows,", train_pool.num_col(), "columns")
print("X_train SHAPE:", X_train.shape)
print(X_train.columns)
print(X_train.dtypes)
'''

train_pool = Pool(X_train, y_train, cat_features=cat_idx)
test_pool = Pool(X_test, y_test, cat_features=cat_idx)

model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='Accuracy',
    iterations=2000,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100,
    od_type='Iter',
    od_wait=100,
    thread_count=4
)

model.fit(train_pool, eval_set=test_pool)

y_pred = model.predict(X_test).reshape(-1)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))