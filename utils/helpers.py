import numpy as np

def collect_data(generator):
    X_all, Y_all = [], []
    for X_batch, Y_batch in generator:
        X_all.append(X_batch)
        Y_all.append(Y_batch)
    return np.vstack(X_all), np.hstack(Y_all)
