import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import new_lib as nl

def acquire_zillow():
    if os.path.isfile('zillow.csv'):
        zil = pd.read_csv('zillow.csv', index_col= 0)
    else:
        zil = nl.connect('zillow', 'zillow.csv', 'SELECT * FROM properties_2017 WHERE propertylandusetypeid = 261')
    return zil

def prep_zillow(zil):
    zil = zil.rename(columns = {'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 
                            'calculatedfinishedsquarefeet': 'square_footage', 
                            'taxvaluedollarcnt': 'tax_value', 'taxamount': 'tax'})
    zil = zil[['id', 'bedrooms', 'bathrooms', 'square_footage', 'tax_value', 'yearbuilt', 
           'tax', 'fips']]
    # getting dataframe into the right subset
    zil = zil.dropna()
    zil = zil.reset_index()
    zil = zil.drop(columns = 'index')
    zil.yearbuilt = zil.yearbuilt.astype(int)
    zil.index.name = 'index'
    return zil
    # dropping all null values and cleaning new dataframe into a more functional dataset

def wrangle_zillow():
    zil = prep_zillow(acquire_zillow())
    return zil

def scale_splits(X_train, X_val, X_test, scaler, columns = False):
    '''
    Accepts input of a train validate test split and a specific scaler. The function will then scale
    the data according to the scaler used and output the splits as scaled splits
    If you want to scale by specific columns enter them in brackets and quotations after entering scaler
    otherwise the function will scale the entire dataframe
    '''
    if columns:
        scale = scaler.fit(X_train[columns])
        train_initial = pd.DataFrame(scale.transform(X_train[columns]),
        columns= X_train[columns].columns.values).set_index([X_train.index.values])
        val_initial = pd.DataFrame(scale.transform(X_val[columns]),
        columns= X_val[columns].columns.values).set_index([X_val.index.values])
        test_initial = pd.DataFrame(scale.transform(X_test[columns]),
        columns= X_test[columns].columns.values).set_index([X_test.index.values])
        train_scaled = X_train.copy()
        val_scaled = X_val.copy()
        test_scaled = X_test.copy()
        train_scaled.update(train_initial)
        val_scaled.update(val_initial)
        test_scaled.update(test_initial)

    else:
        scale = scaler.fit(X_train)
        train_scaled = pd.DataFrame(scale.transform(X_train),
        columns= X_train.columns.values).set_index([X_train.index.values])
        val_scaled = pd.DataFrame(scale.transform(X_val),
        columns= X_val.columns.values).set_index([X_val.index.values])
        test_scaled = pd.DataFrame(scale.transform(X_test),
        columns= X_test.columns.values).set_index([X_test.index.values])
    return train_scaled, val_scaled, test_scaled


