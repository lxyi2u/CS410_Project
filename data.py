import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 归一化
def maxmin_norm(array):
    maxcols = array.max(axis = 0)
    mincols = array.min(axis = 0)
    data_shape = array.shape
    data_rows, data_cols = data_shape
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t

def DataLoader():
    df = pd.read_csv('data.csv')
    df_rows, df_cols = df.shape
    feature = df.drop(['midPrice', 'UpdateTime', 'UpdateMillisec'], axis=1)
    price = df['midPrice']

    predict_days = 10
    window_len = 50

    feature_normal = maxmin_norm(feature.values)

    # 获取label最值
    label = []
    for i in range(df_rows - predict_days):
        label.append(price[i + predict_days] - price[i])
    label_max = max(label)
    label_min = min(label)

    # 跳点判定条件
    df['UpdateTime'] = pd.to_datetime(df.UpdateTime, format='%H:%M:%S')
    time_delta = pd.to_datetime('01:00:00', format='%H:%M:%S') - pd.to_datetime('00:00:00', format='%H:%M:%S')


    # 生成数据集
    X = []
    Y = []
    sample_window = []
    for i in range(df_rows - predict_days):
        sample_window.append(feature_normal[i])
        if len(sample_window) == window_len:
            raise_10days = (float(price[i + predict_days] - price[i] - label_min) / (label_max - label_min))
            X.append(sample_window)
            Y.append(raise_10days)
            sample_window = sample_window[1:]
        # 排除跳点
        if (df['UpdateTime'][i + 1] - df['UpdateTime'][i]) > time_delta:
            sample_window = []
            print("跳点")

    train_num = int(len(X) * 0.7)
    trainX = np.array(X[:train_num])
    testX = np.array(X[train_num:])
    trainY = np.array(Y[:train_num])
    testY = np.array(Y[train_num:])

    print(trainX.shape, testY.shape)

    return trainX, trainY, testX, testY



if __name__ == "__main__":
    DataLoader()