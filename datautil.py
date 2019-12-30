import pandas as pd
import numpy as np
import h5py
import os
import tensorflow as tf
import keras
import random

class DataGenerator(keras.utils.Sequence):

    def __init__(self, datafile, predict_days, window_len, batch_size, dataset):
        self.data = pd.read_csv(datafile)
        count, _ = self.data.shape
        print('count:', count)

        # 获取特征归一化参数
        total_feature = self.data.drop(
            ['midPrice', 'UpdateTime', 'UpdateMillisec'], axis=1)
        self.maxcols = total_feature.values.max(axis=0)
        self.mincols = total_feature.values.min(axis=0)
        self.meancols = total_feature.values.mean(axis=0)
        self.stdcols = total_feature.values.std(axis=0)

        # 获取label归一化参数
        total_price = self.data['midPrice']
        total_label = []
        for i in range(self.data.shape[0] - predict_days):
            total_label.append(
                total_price[i+predict_days] - total_price[i])
        self.label_max = max(total_label)
        self.label_min = min(total_label)
        self.label_mean = np.array(total_label).mean()
        self.label_std = np.array(total_label).std()

        if dataset == 'train':
            self.df = self.data[:int(0.6 * count)]
            self.begin = 0
        elif dataset == 'test':
            self.df = self.data[int(0.7*count):]
            self.begin = int(0.7*count)
        elif dataset == 'validate':
            self.df = self.data[int(0.6*count):int(0.7*count)]
            self.begin = int(0.6*count)

        self.predict_days = predict_days
        self.window_len = window_len
        self.batch_size = batch_size

        self.df_rows, self.df_cols = self.df.shape

        # 获取所选数据集范围内的label（最后10个除外）（为当前位置10条之后的涨跌幅）
        self.price = self.df['midPrice']
        self.label = []
        for i in range(self.df_rows - self.predict_days):
            self.label.append(
                self.price[self.begin+i + self.predict_days] - self.price[self.begin+i])

        # 找出跳点并去除受跳点影响的label
        self.jump_points = []
        self.jump_points_num = 0
        self.df['UpdateTime'] = pd.to_datetime(
            self.df.UpdateTime, format='%H:%M:%S')
        time_delta = pd.to_datetime(
            '01:00:00', format='%H:%M:%S') - pd.to_datetime('00:00:00', format='%H:%M:%S')
        for i in range(self.df_rows-1):
            if (((self.df['UpdateTime'][self.begin+i+1]-self.df['UpdateTime'][self.begin+i]) > time_delta) or
                    ((self.df['UpdateTime'][self.begin+i]-self.df['UpdateTime'][self.begin+i+1]) > time_delta)):
                self.jump_points.append(i)
                self.jump_points_num += 1

        print('jump_points_num:', self.jump_points_num)

        # 总数还要减去跳点造成失效的数据，且知跳点间隔足够大
        self.num = self.df_rows-self.predict_days-self.window_len + \
            1-self.jump_points_num*(self.window_len+self.predict_days-1)

        # 归一化labels
        self.label = self.zscore_normalize_label(self.label)

        # normalize feature
        self.feature = self.df.drop(
            ['midPrice', 'UpdateTime', 'UpdateMillisec'], axis=1)
        self.feature_normal = self.zscore_norm_feature(self.feature.values)
        print('num:', self.num)
        print('label:', len(self.label))

    def __len__(self):
        return np.floor(self.num/self.batch_size-1).astype(np.int)

    def get_num(self):
        return self.num

    def get_len(self):
        return np.floor(self.num/self.batch_size-1).astype(np.int)

    # 计算某窗口因跳点应向后滑动的窗口数
    # 传入原窗口的右端点，返回这个窗口因为跳点的存在需要向后滑动的窗口数
    # 第一次算出原窗口右端点之前的跳点数，窗口向后滑动对应窗口数
    # 但新的滑动也可能经过新的跳点，所以用while直到新的滑动没有再遇到跳点就停止
    def get_prev_jump_num(self, idx):
        prev_jump_num = 0
        while True:
            temp = 0
            for point in self.jump_points:
                if point < idx:
                    temp += 1
            temp -= prev_jump_num
            if temp == 0:
                break
            else:
                prev_jump_num += temp
                idx += temp * (self.window_len+self.predict_days-1)
        return prev_jump_num

    def __getitem__(self, idx):
        print('index', idx)
        batch_x = []
        batch_y = []
        for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
            # 计算窗口因跳点应向后滑动的窗口数
            prev_jump_num = self.get_prev_jump_num(i+self.window_len+self.predict_days-1)
            # 根据此窗口前的跳点数向后滑动窗口
            i += (self.window_len+self.predict_days-1)*prev_jump_num
            batch_x.append(self.feature_normal[i:i+self.window_len])
            batch_y.append(self.label[i+self.window_len-1])

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        # batch_y = np.array(self.label[idx*self.batch_size:(idx+1)*self.batch_size])

        return batch_x, batch_y

    def maxmin_normalize_label(self, labels):
        labels = [(l-self.label_min)/(self.label_max-self.label_min)
                      for l in labels]
        return labels

    def maxmin_denormalize_label(self, labels):
        labels = [l*(self.label_max-self.label_min)+self.label_min for l  in labels]
        return labels

    def zscore_normalize_label(self, labels):
        labels = [(l - self.label_mean) / self.label_std for l in labels]
        return labels

    def zscore_denormalize_label(self, labels):
        labels = [l*self.label_std+self.label_mean for l in labels]
        return labels

    def maxmin_norm_feature(self, array):
        data_shape = array.shape
        data_rows, data_cols = data_shape
        t = np.empty((data_rows, data_cols))
        for i in range(data_cols):
            t[:, i] = (array[:, i] - self.mincols[i]) / (self.maxcols[i] - self.mincols[i])
        return t

    def zscore_norm_feature(self, array):
        data_shape = array.shape
        data_rows, data_cols = data_shape
        t = np.empty((data_rows, data_cols))
        for i in range(data_cols):
            t[:, i] = (array[:, i] - self.meancols[i]) / self.stdcols[i]
        return t

    def get_labels(self):
        return self.label


class DataCertainIntervalGenerator(keras.utils.Sequence):

    def __init__(
            self, datafile, predict_days, window_len,
            interval, batch_size, dataset):
        self.data = pd.read_csv(datafile)
        count, _ = self.data.shape
        print('count', count)

        # 获取特征归一化参数
        total_feature = self.data.drop(
            ['midPrice', 'UpdateTime', 'UpdateMillisec'], axis=1)
        self.maxcols = total_feature.values.max(axis=0)
        self.mincols = total_feature.values.min(axis=0)
        self.meancols = total_feature.values.mean(axis=0)
        self.stdcols = total_feature.values.std(axis=0)

        # 获取label归一化参数
        total_price = self.data['midPrice']
        total_label = []
        for i in range(self.data.shape[0] - predict_days):
            total_label.append(
                total_price[i + predict_days] - total_price[i])
        self.label_max = max(total_label)
        self.label_min = min(total_label)
        self.label_mean = np.array(total_label).mean()
        self.label_std = np.array(total_label).std()

        if dataset == 'train':
            self.df = self.data[:int(0.6 * count)]
            self.begin = 0
        elif dataset == 'test':
            self.df = self.data[int(0.7*count):]
            self.begin = int(0.7*count)
        elif dataset == 'validate':
            self.df = self.data[int(0.6*count):int(0.7*count)]
            self.begin = int(0.6*count)

        self.predict_days = predict_days
        self.window_len = window_len
        self.interval = interval
        self.batch_size = batch_size

        self.df_rows, self.df_cols = self.df.shape

        self.total_num = self.df_rows - self.predict_days - self.window_len + 1
        self.num = int((self.total_num-1)/self.interval) + 1

        # get label
        self.price = self.df['midPrice']
        self.label = []

        for i in range(self.df_rows - self.predict_days):
            self.label.append(
                self.price[self.begin + i + self.predict_days] - self.price[self.begin + i])
        self.label = self.zscore_normalize_label(self.label)

        # normalize feature
        self.feature = self.df.drop(
            ['midPrice', 'UpdateTime', 'UpdateMillisec'], axis=1)
        self.feature_normal = self.zscore_norm_feature(self.feature.values)
        print('num:', self.num)
        print('label:', len(self.label))

    def __len__(self):
        return np.floor(self.num/self.batch_size-1).astype(np.int)

    def get_num(self):
        return self.num

    def get_len(self):
        return np.floor(self.num / self.batch_size - 1).astype(np.int)

    def __getitem__(self, idx):
        print('index', idx)
        batch_x = []
        batch_y = []
        for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
            right_boundary = i*self.interval
            left_boundary = max(0, right_boundary - self.interval + 1)
            index = random.choice(range(left_boundary, right_boundary+1))
            batch_x.append(self.feature_normal[index:index+self.window_len])
            batch_y.append(self.label[index+self.window_len-1])

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        return batch_x, batch_y

    def maxmin_normalize_label(self, labels):
        labels = [(l-self.label_min)/(self.label_max-self.label_min)
                      for l in labels]
        return labels

    def maxmin_denormalize_label(self, labels):
        labels = [l*(self.label_max-self.label_min)+self.label_min for l  in labels]
        return labels

    def zscore_normalize_label(self, labels):
        labels = [(l - self.label_mean) / self.label_std for l in labels]
        return labels

    def zscore_denormalize_label(self, labels):
        labels = [l*self.label_std+self.label_mean for l in labels]
        return labels

    def maxmin_norm_feature(self, array):
        data_shape = array.shape
        data_rows, data_cols = data_shape
        t = np.empty((data_rows, data_cols))
        for i in range(data_cols):
            t[:, i] = (array[:, i] - self.mincols[i]) / (self.maxcols[i] - self.mincols[i])
        return t

    def zscore_norm_feature(self, array):
        data_shape = array.shape
        data_rows, data_cols = data_shape
        t = np.empty((data_rows, data_cols))
        for i in range(data_cols):
            t[:, i] = (array[:, i] - self.meancols[i]) / self.stdcols[i]
        return t

class IdentityDataGenerator(keras.utils.Sequence):

    def __init__(self, datafile, batch_size, dataset):
        self.data = pd.read_csv(datafile)
        count, _ = self.data.shape
        print('count:', count)

        # 获取特征归一化参数
        total_feature = self.data.drop(
            ['midPrice', 'UpdateTime', 'UpdateMillisec'], axis=1)
        self.maxcols = total_feature.values.max(axis=0)
        self.mincols = total_feature.values.min(axis=0)
        self.meancols = total_feature.values.mean(axis=0)
        self.stdcols = total_feature.values.std(axis=0)

        if dataset == 'train':
            self.df = self.data[:int(0.6 * count)]
        elif dataset == 'test':
            self.df = self.data[int(0.7 * count):]
        elif dataset == 'validate':
            self.df = self.data[int(0.6 * count):int(0.7 * count)]

        self.batch_size = batch_size

        self.df_rows, self.df_cols = self.df.shape
        self.num = self.df_rows

        # normalize feature
        self.feature = self.df.drop(
            ['midPrice', 'UpdateTime', 'UpdateMillisec'], axis=1)
        self.feature_normal = self.zscore_norm_feature(self.feature.values)
        print('num:', self.num)

    def __len__(self):
        return np.floor(self.num/self.batch_size).astype(np.int)

    def __getitem__(self, idx):
        batch = []
        for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
            batch.append(
                self.feature_normal[i])

        batch = np.array(batch)
        # print('batch_x:', batch_x.shape)

        return batch, batch

    def get_len(self):
        return self.df.shape[0]

    def maxmin_norm_feature(self, array):
        data_shape = array.shape
        data_rows, data_cols = data_shape
        t = np.empty((data_rows, data_cols))
        for i in range(data_cols):
            t[:, i] = (array[:, i] - self.mincols[i]) / (self.maxcols[i] - self.mincols[i])
        return t

    def zscore_norm_feature(self, array):
        data_shape = array.shape
        data_rows, data_cols = data_shape
        t = np.empty((data_rows, data_cols))
        for i in range(data_cols):
            t[:, i] = (array[:, i] - self.meancols[i]) / self.stdcols[i]
        return t


class IdentityDataReader(keras.utils.Sequence):
    def __init__(self, datafile, dataset):
        self.data = pd.read_csv(datafile)
        count, _ = self.data.shape
        print('count:', count)

        # 获取特征归一化参数
        total_feature = self.data.drop(
            ['midPrice', 'UpdateTime', 'UpdateMillisec'], axis=1)
        self.maxcols = total_feature.values.max(axis=0)
        self.mincols = total_feature.values.min(axis=0)
        self.meancols = total_feature.values.mean(axis=0)
        self.stdcols = total_feature.values.std(axis=0)

        if dataset == 'train':
            self.df = self.data[:int(0.6 * count)]
        elif dataset == 'test':
            self.df = self.data[int(0.7 * count):]
        elif dataset == 'validate':
            self.df = self.data[int(0.6 * count):int(0.7 * count)]

        self.df_rows, self.df_cols = self.df.shape
        self.num = self.df_rows
        print('number:', self.num)

        # normalize feature
        self.feature = self.df.drop(
            ['midPrice', 'UpdateTime', 'UpdateMillisec'], axis=1)
        self.feature_normal = self.zscore_norm_feature(self.feature.values)

    def get_data(self):
        return self.feature_normal

    def maxmin_norm_feature(self, array):
        data_shape = array.shape
        data_rows, data_cols = data_shape
        t = np.empty((data_rows, data_cols))
        for i in range(data_cols):
            t[:, i] = (array[:, i] - self.mincols[i]) / (self.maxcols[i] - self.mincols[i])
        return t

    def zscore_norm_feature(self, array):
        data_shape = array.shape
        data_rows, data_cols = data_shape
        t = np.empty((data_rows, data_cols))
        for i in range(data_cols):
            t[:, i] = (array[:, i] - self.meancols[i]) / self.stdcols[i]


if __name__ == "__main__":
    DataGenerator('./dataset/data.csv', 10, 50, 32, 'test')
