import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import metrics, feature_selection
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from itertools import product
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier




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

# Plotting Decision Regions
labels = ['Logistic Regression',
          'Random Forest',
          'RBF kernel SVM',
          'Ensemble']


clf1 = KNeighborsClassifier(n_neighbors=7) #strong
clf2 = SVC(random_state=0, kernel='rbf', probability=True) #strong
clf3 = SVC(gamma=2, C=1, probability=True) #strong
clf4 = RandomForestClassifier(random_state=0) #strong


estimators = [
    ('knn', clf1), ('svm', clf2),
    ('svm_gamma', clf3), ('rf', clf4), 
]

eclf = VotingClassifier(estimators=estimators, voting='soft')

params = {
    'rf__n_estimators': [20, 50, 100],
    'knn__n_neighbors': [2,3,4,5,6,7,8,9,10,11,12],
}

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)

for label, clf in estimators:
    print(label)
    if label == 'knn':
        params = {"n_neighbors": [2,3,4,5,6,7,8,9,10,11,12]}
        clf = GridSearchCV(estimator=clf, param_grid=params, cv=5)
    model = clf.fit(X_train, y_train)
    print("Mean squared error: %.2f"
          % metrics.mean_squared_error(y_test, model.predict(X_test)))
    # Explained variance score: 1 is perfect prediction
    print("R^2", metrics.r2_score(y_test, model.predict(X_test)))
    scores = cross_val_score(model, X_train, y_train)
    print("Cross validation ave score", scores.mean())

print("Ensemble")
model = grid.fit(X_train, y_train)
print("Mean squared error: %.2f"
      % metrics.mean_squared_error(y_test, model.predict(X_test)))
# Explained variance score: 1 is perfect prediction
print("R^2", metrics.r2_score(y_test, model.predict(X_test)))
scores = cross_val_score(model, X_train, y_train)
print("Cross validation ave score", scores.mean())


    
