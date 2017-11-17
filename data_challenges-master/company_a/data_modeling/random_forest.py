import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics, feature_selection
from sklearn.svm import SVC
from sklearn import tree
import pandas as pd
import pydotplus 

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def to_numeric(df):
    if "CustomerID" in df.columns:
        del df["CustomerID"]
    df["YearsAtCurrentEmployer"] = df["YearsAtCurrentEmployer"].replace("10+","10")
    df["YearsAtCurrentEmployer"] = df["YearsAtCurrentEmployer"].astype(float)
    for column in df.columns:
        if not (df[column].dtype == np.float64 or df[column].dtype == np.int64):
            tmp_df = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, tmp_df], axis=1)
            del df[column]
    return df


# Initializing Classifiers
df = pd.read_csv("final_resampled_data.csv")
df = to_numeric(df)
columns = [
    'CheckingAccountBalance_debt',
    'CheckingAccountBalance_none',
    'CheckingAccountBalance_some',
    'Co-Applicant_co-app',
    'Co-Applicant_guarant',
    'DebtsPaid_delayed',
    'DebtsPaid_paid',
    'LoanReason_busin',
    'LoanReason_goods',
    'RentOrOwnHome_free',
    'RentOrOwnHome_owned',
    'RequestedAmount',
    'SavingsAccountBalance_high',
    'SavingsAccountBalance_none',
    'SavingsAccountBalance_some',
    'YearsAtCurrentEmployer'
]

X = df[columns]
size = len(X)
train_size = int(size * 0.8)
# Split the data into training/testing sets

X_train = X[:train_size]
X_test = X[train_size:]


y = df["WasTheLoanApproved_Y"]
# Split the targets into training/testing sets
y_train = y[:train_size]
y_test = y[train_size:]

param_grid = {
    "n_estimators": [20, 40, 100, 120],
    "max_depth": [30, 40],
    "min_samples_split": [2, 3, 10],
    "min_samples_leaf": [2, 3, 10],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
}

clf = RandomForestClassifier()
model = GridSearchCV(clf, param_grid=param_grid)
model.fit(X_train, y_train)

print("Random Forest precision:")
print("Mean squared error: %.2f"
      % metrics.mean_squared_error(y_test, model.predict(X_test)))
# Explained variance score: 1 is perfect prediction
print("R^2", metrics.r2_score(y_test, model.predict(X_test)))
scores = cross_val_score(model, X_train, y_train)
print("Cross validation ave score", scores.mean())

