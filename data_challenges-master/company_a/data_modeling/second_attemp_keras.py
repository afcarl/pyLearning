import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def to_numeric(df):
    numeric_columns = []
    for column in df.columns:
        if not (df[column].dtype == np.float64 or df[column].dtype == np.int64):
            tmp_df = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, tmp_df], axis=1)
            del df[column]
        else:
            numeric_columns.append(column)
    return df, numeric_columns

def feature_transform(df, col_name):
    df[col_name+"_squared"] = df[col_name].apply(lambda x: x*x)
    df[col_name+"_log"] = df[col_name].apply(lambda x: math.log(x) if x != 0 else x)
    df[col_name+"_sin"] = df[col_name].apply(lambda x: math.sin(x))
    df[col_name+"_sqrt"] = df[col_name].apply(lambda x: math.sqrt(x))
    try:
        df[col_name+"_gamma"] = df[col_name].apply(lambda x: math.gamma(x) if x!=0 else x)
    except OverflowError:
        pass
    return df

def data():
                                                 
    df = pd.read_csv("final_resampled_data.csv")

    columns = [
        'CheckingAccountBalance',
        'Co-Applicant',
        'DebtsPaid',
        'LoanReason',
        'RentOrOwnHome',
        'RequestedAmount',
        'SavingsAccountBalance',
        'Age'
    ]

    x = df[columns]
    x, numeric_cols = to_numeric(x)
    for col in numeric_cols:
        x = feature_transform(x, col)
    size = len(x)
    train_size = int(size * 0.9)
    # Split the data into training/testing sets

    x_train = x[:train_size]
    x_test = x[train_size:]

    y = pd.get_dummies(df["WasTheLoanApproved"], prefix="WasTheLoanApproved")
    y = y["WasTheLoanApproved_Y"]
    # Split the targets into training/testing sets
    y_train = y[:train_size]
    y_test = y[train_size:]

    x_train = x_train.as_matrix()
    x_test = x_test.as_matrix()
    return x_train, y_train, x_test, y_test

# larger model
def create_network():
    # create model
    model = Sequential()
    model.add(Dense(34, input_dim=34, kernel_initializer='normal', activation='relu'))
    model.add(Dense(17, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#load data
x_train, y_train, x_test, y_test = data()
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_network, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, x_train, y_train, cv=kfold)
print("Neural Network: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

