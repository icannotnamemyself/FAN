import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rse(y_true, y_pred):
    # mtgnn 中使用的rse ， 分母除了 标准差
    mse = mean_squared_error(y_true, y_pred)
    rse = np.sqrt(mse) /( np.std(y_true)  ) # *np.sqrt((n-1)/n) )
    return rse


def rae(y_true, y_pred):
    mse = mean_absolute_error(y_true, y_pred)
    rse = np.sqrt(mse) / np.std(y_true) 
    return rse

