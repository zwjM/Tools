from mimetypes import init
from os import lstat
from statistics import mean
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
#01 把DataFrame中日频的列通过求平均值的方式转化为月频
def dayToMonth(data,col):
    data.date = pd.to_datetime(data.date)
    data['date'] = data['date'].dt.strftime('%Y-%m')
    data = data.groupby('date').mean()[[col]]
    data.columns = ['data']
    return data
#02 对于sklearn中的机器学习模型绘制变量的重要性（变量的coef或者feature_importances_）
def plot_importance(model, x_train):
    if hasattr(model,'coef_'):
        coefs = pd.DataFrame(model.coef_, x_train.columns)
    else:
        coefs = pd.DataFrame(model.feature_importances_, x_train.columns)
    coefs.columns = ["coefs"]
    coefs["coefs_abs"] = coefs.coefs.apply(np.abs)
    coefs = coefs.sort_values(by="coefs_abs", ascending=False).drop(["coefs_abs"], axis=1)
    plt.figure(figsize=(16, 6))
    coefs.coefs.plot(kind="bar")
    plt.grid(True, axis="y")
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles="dashed")
    plt.show()


        

#03 convert series to supervised learning 得到训练的数据格式
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    @n_in :获取t-1到t-n的特征
    @n_out :获取t+1 到t+n 的特征

    """
    n_vars = 1 if type(data) is list else data.shape[1]
    raw_columns = data.columns.values
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (raw_columns[j%n_vars], i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (raw_columns[j%n_vars])) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (raw_columns[j%n_vars], i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


