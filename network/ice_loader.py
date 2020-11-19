import pandas as pd
import numpy as np
from numpy import array

import os

from numpy import concatenate



def getdata(filename):
    dataset = np.genfromtxt(filename, delimiter=',',dtype="float32")
    dataset = dataset[:, 0:-2]
    return dataset


def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y,D = list(), list(),list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y,seq_d = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :],sequences[end_ix,:]
        X.append(seq_x)
        D.append(seq_d)
        y.append(seq_y)
    return array(X), array(D),array(y)


def train_test(data_dir,n_steps_in,n_steps_out):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_data = getdata(train_path)
    test_data = getdata(test_path)
    # sc_X = MinMaxScaler(feature_range=(0, 1))
    # training_set_scaled = sc_X.fit_transform(training_set)

    train_X,train_d, train_y = split_sequences(train_data, n_steps_in, n_steps_out)
    test_X,test_d,test_y = split_sequences(test_data, n_steps_in, n_steps_out)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_d,train_y, test_X,test_d ,test_y





