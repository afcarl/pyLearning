import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import tree
import pandas as pd
from sklearn.model_selection import cross_val_score

def to_numeric(df):
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
svm_clf = SVC(random_state=0, probability=True)

df = pd.read_csv("final_resampled_data.csv")
df = to_numeric(df)
columns = df.columns.tolist()
[columns.remove(elem) for elem in
 ['WasTheLoanApproved_N', 'WasTheLoanApproved_Y']]

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

model = svm_clf.fit(X_train, y_train)
y_pred = model.predict(X_train)
tree_clf = tree.DecisionTreeClassifier()
tree_model = tree_clf.fit(X_train, y_pred)

print("Mean squared error: %.2f"
      % metrics.mean_squared_error(y_test, tree_model.predict(X_test)))
# Explained variance score: 1 is perfect prediction
print("R^2", metrics.r2_score(y_test, tree_model.predict(X_test)))
scores = cross_val_score(tree_model, X_train, y_train)
print("Cross validation ave score", scores.mean())
