import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import random
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras import regularizers
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.layers.normalization import BatchNormalization
import math

seed = 7
np.random.seed(seed)


def data():

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

    # encode class values as integers
    def one_hot_encoding(df):
        encoder = LabelEncoder()
        encoder.fit(df)
        encoded_df = encoder.transform(df)
        # convert integers to dummy variables (i.e. one hot encoded)
        return np_utils.to_categorical(encoded_df)

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

def model(x_train, y_train, x_test, y_test):

    # create model
    model = Sequential()
    # layer 1
    model.add(Dense(
        1024,
        input_shape=(34,),
        kernel_regularizer=regularizers.l2(0.1),
        activity_regularizer=regularizers.l1(0.1)
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # layer 2
    model.add(Dense(
        1024,
        input_shape=(1024,),
        kernel_regularizer=regularizers.l2(0.1),
        activity_regularizer=regularizers.l1(0.1)
    ))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense(
        512,
        input_shape=(1024,),
        kernel_regularizer=regularizers.l2(0.1),
        activity_regularizer=regularizers.l1(0.1)
    ))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense(
        256,
        input_shape=(512,),
        kernel_regularizer=regularizers.l2(0.1),
        activity_regularizer=regularizers.l1(0.1)
    ))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense(
        128,
        input_shape=(256,),
        kernel_regularizer=regularizers.l2(0.1),
        activity_regularizer=regularizers.l1(0.1)
    ))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense(
        64,
        input_shape=(128,),
        kernel_regularizer=regularizers.l2(0.1),
        activity_regularizer=regularizers.l1(0.1)
    ))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense(
        32,
        input_shape=(64,),
        kernel_regularizer=regularizers.l2(0.1),
        activity_regularizer=regularizers.l1(0.1)
    ))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense(
        16,
        input_shape=(32,),
        kernel_regularizer=regularizers.l2(0.1),
        activity_regularizer=regularizers.l1(0.1)
    ))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense(
        8,
        input_shape=(16,),
        kernel_regularizer=regularizers.l2(0.1),
        activity_regularizer=regularizers.l1(0.1)
    ))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    model.add(Dropout({{uniform(0.5, 1)}}))
    model.add(Dense(
        4,
        input_shape=(8,),
        kernel_regularizer=regularizers.l2(0.1),
        activity_regularizer=regularizers.l1(0.1)
    ))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    model.add(Dropout({{uniform(0.5, 1)}}))

    # layer 5
    model.add(Dense(
        2,
        input_shape=(4,),
        kernel_regularizer=regularizers.l2(0.1),
        activity_regularizer=regularizers.l1(0.1)
    ))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    model.compile(loss='binary_crossentropy',
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=500,
              epochs=2,
              verbose=1,
              validation_data=(x_test, y_test))

    score, acc = model.evaluate(x_test, y_test, verbose=0)
    return {"loss": acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=3,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print()
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print()
    print(best_run)
