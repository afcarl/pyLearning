from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import numpy as np
import pandas as pd
from sklearn import model_selection
from scipy import stats 

def data():
    """
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    def corr_categorical_with_wages(df, wages):
        correlations = {}
        columns = df.columns.tolist()
        for col in columns:
            correlations["wages" + '___' + col] = stats.pointbiserialr(wages, df[col].values)
        results = pd.DataFrame.from_dict(correlations, orient="index")
        results.columns = ["correlation", "pvalues"]
        results.sort_index(inplace=True)
        return results

    def corr_numeric_with_wages(df, wages):
        correlations = {}
        columns = df.columns.tolist()
        start = time.time()
        for col in columns:
            correlations["wages" + '___' + col] = stats.spearmanr(wages, df[col].values)
        results = pd.DataFrame.from_dict(correlations, orient="index")
        results.columns = ["correlation", "pvalues"]
        results.sort_index(inplace=True)
        return results

    def numeric_only_data(df):
        numeric_columns = []
        for column in df.columns:
            if df[column].dtype == np.float64 or df[column].dtype == np.int64:
                numeric_columns.append(column)
        return numeric_columns

    def process_index(i):
        return i.replace("wages___","")

    df = pd.read_csv("1hb_2014.csv")

    tmp_df = df[df["visa_class"] == "H-1B"]
    tmp_df.dropna(inplace=True)
    categorical_columns = ["status", "lca_case_employer_state", "lca_case_soc_name", "lca_case_wage_rate_unit", "full_time_pos"]
    numeric_columns = numeric_only_data(df)
    columns = df.columns.tolist()
    columns = [column for column in columns if column in numeric_columns]
    categorical_df = pd.get_dummies(tmp_df[categorical_columns])
    numeric_df = tmp_df[numeric_columns]
    cat_corr_df = corr_categorical_with_wages(categorical_df, numeric_df["pw_1"].values)
    num_corr_df = corr_categorical_with_wages(numeric_df, numeric_df["pw_1"].values)
    final_df = pd.concat([numeric_df, categorical_df], axis=1)
    variables_to_remove = []    
    for i in cat_corr_df.index:
        if cat_corr_df.ix[i]["pvalues"] > 0.05:
            variables_to_remove.append(process_index(i))

    for i in num_corr_df.index:
        if num_corr_df.ix[i]["pvalues"] > 0.05:
            variables_to_remove.append(process_index(i))
    cols = final_df.columns.tolist()
    cols = [col for col in cols if col not in variables_to_remove]
    final_df = final_df[cols]
    cols = final_df.columns.tolist()
    cols.remove("pw_1")
    X = final_df[cols]
    y = final_df["pw_1"]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = X_train.as_matrix()
    y_train = y_train.as_matrix()
    X_test = X_test.as_matrix()
    y_test = y_test.as_matrix()
    return X_train, y_train, X_test, y_test


def model(X_train, y_train, X_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense(117, input_dim=117))
    model.add(Activation({{choice(['relu', 'sigmoid', 'tanh', 'hard_sigmoid', 'softsign', 'softplus', 'selu', 'elu', 'softmax','linear'])}}))
    model.add(Dense(58, kernel_initializer='normal', activation='relu'))
    model.add(Dense(28, kernel_initializer='normal', activation='relu'))
    model.add(Dense(14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train,
              batch_size=100,
              epochs=1,
              verbose=2,
              validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, y_train, X_test, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
