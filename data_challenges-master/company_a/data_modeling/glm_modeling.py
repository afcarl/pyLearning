import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics, feature_selection
import statsmodels.api as sm
import pandas as pd

#TODO make a tools section and put this there
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

def sklearn_regression(X_train, X_test, y_train, y_test):
    # Create linear regression object
    regr = linear_model.LogisticRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % metrics.mean_squared_error(y_test, regr.predict(X_test)))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(X_test, y_test))

    print("R^2", metrics.r2_score(y_test, regr.predict(X_test)))

#throw the kitchen sink
def statsmodels_reg_testing_family(X_train, X_test, y_train, y_test):
    families = [
        sm.families.Binomial,
        sm.families.Gamma,
        sm.families.Gaussian,
        sm.families.NegativeBinomial,
        sm.families.Poisson
    ]
    family_names = [
        'Binomial',
        'Gamma',
        'Gaussian',
        'NegativeBinomial',
        'Poisson'
    ]
    for family, family_name in zip(families, family_names):
        print()
        print(family_name)
        model = sm.GLM(y_train, X_train, family=family())
        results = model.fit()
        print("Mean squared error: %.2f"
              % metrics.mean_squared_error(y_test, results.predict(X_test)))
        # Explained variance score: 1 is perfect prediction
        print("R^2", metrics.r2_score(y_test, results.predict(X_test)))
        print(results.summary())
        print()


def statsmodels_reg(X_train, X_test, y_train, y_test):
    model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
    results = model.fit()
    print("Mean squared error: %.2f"
          % metrics.mean_squared_error(y_test, results.predict(X_test)))
    # Explained variance score: 1 is perfect prediction
    print("R^2", metrics.r2_score(y_test, results.predict(X_test)))
    print(results.summary())
    print()


def naive_feature_selection(df):
    """
    This list comes from the set of variables found to be correlated with
    'WasTheLoanApproved_Y'.  I then removed some variables that are correlated with
    each other."""
    variables_of_interest = [
        "Age",
        "CheckingAccountBalance_none",
        "Co-Applicant_co-app",
        "DebtsPaid_paid",
        "LoanPayoffPeriodInMonths",
        "LoanReason_goods",
        "RentOrOwnHome_owned",
        "RequestedAmount",
        "SavingsAccountBalance_none",
        "YearsAtCurrentEmployer",
        "YearsInCurrentResidence"
    ]
    return df[variables_of_interest]


# Load the diabetes dataset
df = pd.read_csv("ds-all-data.csv")
df = to_numeric(df)
columns = df.columns.tolist()

[columns.remove(elem) for elem in
 ['WasTheLoanApproved_1', 'WasTheLoanApproved_N', 'WasTheLoanApproved_Y']]

X = df[columns]
X = naive_feature_selection(X)
size = len(X)
train_size = int(size * 0.8)
# Split the data into training/testing sets

X_train = X[:train_size]
X_test = X[train_size:]


y = df["WasTheLoanApproved_Y"]
# Split the targets into training/testing sets
y_train = y[:train_size]
y_test = y[train_size:]

#sklearn_regression(X_train, X_test, y_train, y_test)
statsmodels_reg(X_train, X_test, y_train, y_test)

