from statistics import mean
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
def dayToMonth(data):
    data.date = pd.to_datetime(data.date)
    data['date'] = data['date'].dt.strftime('%Y-%m')
    data = data.groupby('date').mean()[['close']]
    data.columns = ['data']
    return data

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


# convert series to supervised learning 得到训练的数据格式
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
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


#归一化
def minMaxNorm(train_X,train_Y,val_X,val_Y):
    """
    Dataframe 或者 ndarray都可
    """
    X_std = train_X.std(axis=0)
    Y_std = train_Y.std(axis=0)
    meanX =train_X.mean(axis=0)
    meanY = train_Y.mean(axis=0) 
    X_train_norm = (train_X - meanX) / X_std
    Y_train_norm = (train_Y - meanY) / Y_std
    X_val_norm = (val_X - meanX) / X_std
    Y_val_norm = (val_Y - meanY) / Y_std
    return X_train_norm,Y_train_norm,X_val_norm,Y_val_norm

#define forecasts=[] 在使用之前
def rollWindowArima(sequence,window=15,pred_n=1):
    """sequence:Dataframe,1列
        window:窗口大小
        pred_n:每个窗口向后预测的步长

    """
    
    def calculate(train):
        model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(train)
        forecast = model.predict(n_periods =pred_n)
        global forecasts
        forecasts = np.append(forecasts,forecast)
        return forecast
    sequence.rolling(window,min_periods=window).apply(calculate)

    #作图
    forecasts=pd.DataFrame(forecasts[:-1],index = sequence.index[window:],columns=['Prediction'])
    plt.figure(figsize=(14,8), dpi=800)
    plt.plot(sequence[window:],label='Valid')
    plt.plot(forecasts, label='Prediction')
    plt.legend()
    rmse = mean_squared_error(sequence[window:], forecasts[:-1], squared=False)
    plt.title('RMSE : %.4f' % rmse)
    plt.show()

    return forecasts
