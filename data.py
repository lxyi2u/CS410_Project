import pandas as pd
import numpy as np
import h5py
import os
import tensorflow as tf
import keras
import random

# 获得特征值归一化的参数
def get_feature_normalize(datafile):
    df = pd.read_csv(datafile)
    feature = df.drop(['UpdateTime', 'UpdateMillisec'], axis=1)
    meancols = feature.values.mean(axis=0)
    stdcols = feature.values.std(axis=0)

    return meancols, stdcols

class DataGenerator(keras.utils.Sequence):

    def __init__(self, datafile, predict_days, window_len, batch_size, interval, meancols, stdcols):
        self.df = pd.read_csv(datafile)
        print('shape', self.df.shape)

        self.predict_days = predict_days
        self.window_len = window_len
        self.batch_size = batch_size
        self.interval = interval
        self.meancols = meancols
        self.stdcols = stdcols

        # 获取feature并归一化
        self.feature = self.df.drop(['UpdateTime', 'UpdateMillisec'], axis=1)
        self.feature = self.feature_normalize(self.feature.values)

        # 获取midPrice
        self.price = self.df['midPrice'].values

        # 找出跳点，将数据集分割
        self.features = []
        self.prices = []
        self.jump_points = []
        self.df['UpdateTime'] = pd.to_datetime(self.df.UpdateTime, format='%Y-%m-%d %H:%M:%S')
        time_delta = pd.to_datetime('01:00:00', format='%H:%M:%S') - pd.to_datetime('00:00:00', format='%H:%M:%S')
        previous = 0 # 上一个自数据集的终点，也就是下一个数据集的起点
        for i in range(self.df.shape[0] - 1):
            if (((self.df['UpdateTime'][i+1] - self.df['UpdateTime'][i]) > time_delta) or
                    ((self.df['UpdateTime'][i] - self.df['UpdateTime'][i+1]) > time_delta)):
                self.jump_points.append(i)
                self.features.append(self.feature[previous:i+1])
                self.prices.append(self.price[previous:i+1])
                previous = i+1
        self.features.append((self.feature[previous:]))
        self.prices.append(self.price[previous:])

        # 计算每个子数据集可产生的batch数
        self.set_batch_num = [] # 存放每个子数据集的batch数目
        for feature in self.features:
            num = int((feature.shape[0]-self.predict_days-self.window_len)/self.interval)+1 # 考虑interval后的数据总数
            self.set_batch_num.append(int(num/self.batch_size))
        print('set_batch_num', self.set_batch_num)

        # 计算所有子数据集的batch总数
        self.batch_num = sum(self.set_batch_num)
        print('batch_num', self.batch_num)

    def __len__(self):
        return self.batch_num

    def get_len(self):
        return self.batch_num

    def __getitem__(self, idx):
        print('index', idx)

        # 首先根据idx计算所取的batch应该在哪一个子数据集内, 并直接把应该选的子数据集放入chosen_feature和chosen_price中
        set_index = -1
        for i in range(len(self.set_batch_num)):
            if (idx < self.set_batch_num[i]):
                set_index = i
                break
            else:
                idx -= self.set_batch_num[i]
        if (set_index == -1):
            raise Exception('idx溢出', idx)

        # 根据选出来的子数据集生成batch
        batch_x = []
        batch_y = []
        for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
            right_boundary = i*self.interval
            left_boundary = max(0, right_boundary-self.interval+1)
            index = random.choice(range(left_boundary, right_boundary+1))
            batch_x.append(self.features[set_index][index:index+self.window_len])
            batch_y.append(self.prices[set_index][index+self.window_len-1+self.predict_days]-self.prices[set_index][index+self.window_len-1])
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        return batch_x, batch_y

    # 特征归一化
    def feature_normalize(self, array):
        data_shape = array.shape
        data_rows, data_cols = data_shape
        t = np.empty((data_rows, data_cols))
        for i in range(data_cols):
            t[:, i] = (array[:, i] - self.meancols[i]) / self.stdcols[i]
        return t