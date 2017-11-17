import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model, decomposition, metrics, feature_selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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

logistic = linear_model.LogisticRegression()
lin_reg = linear_model.LinearRegression()
pca = decomposition.PCA()
pipe = Pipeline(steps=[
    ('pca', pca),
    ('logistic', logistic),
    ('linear_reg', lin_reg)
])

df = pd.read_csv("resampled_data.csv")
#df = pd.read_csv("ds-all-data.csv")
#df = to_numeric(df)
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
n_components = [10, 11, 12, 13, 14, 15, 16, 17]
Cs = np.logspace(-4, 4, 3)
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(X_train, y_train)
pca.fit(X_train)
print(estimator.best_estimator_.named_steps['pca'].n_components)
print(pca.explained_variance_)
print("Mean squared error: %.2f"
      % metrics.mean_squared_error(y_test, estimator.predict(X_test)))
# Explained variance score: 1 is perfect prediction
print("R^2", metrics.r2_score(y_test, estimator.predict(X_test)))
