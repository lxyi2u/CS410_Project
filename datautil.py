import pandas as pd
import numpy as np
import h5py
import os
import tensorflow as tf
import keras


class HDF5DatasetWriter:
    def __init__(self, data_dims, labels_dims, outputPath, bufSize=20):
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath' already"
                             "exists and cannot be overwritten. Manually"
                             "delete the file before continuing", outputPath)

        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(
            "data", data_dims, maxshape=(None,) + data_dims[1:], dtype="float")
        self.labels = self.db.create_dataset(
            "labels", labels_dims, maxshape=(None,) + labels_dims[1:], dtype="float")
        self.ddims = data_dims
        self.ldims = labels_dims
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, data, labels):
        # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表)
        # 注意，用extend还有好处，添加的数据不会是之前list的引用！！
        self.buffer["data"].append(data)
        self.buffer["labels"].append(labels)

        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["data"])
        if i > self.data.shape[0]:
            # 拓展大小
            new_data_shape = (self.data.shape[0] * 2,) + self.ddims[1:]
            new_label_shape = (self.labels.shape[0] * 2,) + self.ldims[1:]
            print("resize to new_shape:", new_data_shape)
            self.data.resize(new_data_shape)
            self.labels.resize(new_label_shape)
        self.data[self.idx:i, :, :] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        print("h5py have writen %d data" % i)
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()


# 归一化
def maxmin_norm(array):
    maxcols = array.max(axis=0)
    mincols = array.min(axis=0)
    data_shape = array.shape
    data_rows, data_cols = data_shape
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t

# 将数据集写入train.h5和test.h5,但文件较大


def DataH5Writer(datafile, predict_days, window_len):
    df = pd.read_csv(datafile)
    print('read in succeed')
    df_rows, df_cols = df.shape
    train_num = int(df_rows*0.7)
    feature = df.drop(['midPrice', 'UpdateTime', 'UpdateMillisec'], axis=1)
    price = df['midPrice']

    feature_normal = maxmin_norm(feature.values)

    trainWriter = HDF5DatasetWriter(
        (100, window_len, feature.shape[1]), (100,), './train')
    testWrite = HDF5DatasetWriter(
        (100, window_len, feature.shape[1]), (100,), './test')

    # 获取label最值
    label = []
    for i in range(df_rows - predict_days):
        label.append(price[i + predict_days] - price[i])
    label_max = max(label)
    label_min = min(label)

    # 跳点判定条件
    df['UpdateTime'] = pd.to_datetime(df.UpdateTime, format='%H:%M:%S')
    time_delta = pd.to_datetime(
        '01:00:00', format='%H:%M:%S') - pd.to_datetime('00:00:00', format='%H:%M:%S')

    # 生成数据集
    sample_window = []
    for i in range(df_rows - predict_days):
        sample_window.append(feature_normal[i])
        if len(sample_window) == window_len:
            raise_10days = (float(price[i + predict_days] - price[i]
                                  - label_min) / (label_max - label_min))
            if i <= train_num:
                trainWriter.add(sample_window, raise_10days)
            else:
                testWrite.add(sample_window, raise_10days)
            sample_window = sample_window[1:]
        # 排除跳点
        if (df['UpdateTime'][i + 1] - df['UpdateTime'][i]) > time_delta:
            sample_window = []
            # print("跳点")

    trainWriter.close()
    testWrite.close()


def load_dataset(datafile, predict_days, window_len):
    df = pd.read_csv(datafile)
    print('read in succeed')
    df_rows, df_cols = df.shape
    feature = df.drop(['midPrice', 'UpdateTime', 'UpdateMillisec'], axis=1)
    price = df['midPrice']

    train_num = int(df_rows * 0.7)
    feature_normal = maxmin_norm(feature.values)

    # 获取label最值
    label = []
    for i in range(df_rows - predict_days):
        label.append(price[i + predict_days] - price[i])
    label_max = max(label)
    label_min = min(label)

    # 跳点判定条件
    df['UpdateTime'] = pd.to_datetime(df.UpdateTime, format='%H:%M:%S')
    time_delta = pd.to_datetime(
        '01:00:00', format='%H:%M:%S') - pd.to_datetime('00:00:00', format='%H:%M:%S')

    def train_generator(predict_days, window_len):
        train_sample_window = []
        for i in range(train_num):
            print('round:', i)
            train_sample_window.append(feature_normal[i])
            if len(train_sample_window) == window_len:
                raise_10days = (float(price[i + predict_days] - price[i]
                                      - label_min) / (label_max - label_min))
                data = np.array(train_sample_window)
                label = np.array(raise_10days)
                print(data.shape)
                print(label.shape)
                yield data, label
                train_sample_window = train_sample_window[1:]
            # 排除跳点
            if (df['UpdateTime'][i + 1] - df['UpdateTime'][i]) > time_delta:
                train_sample_window = []
                # print("jump point")

    def test_generator(predict_days, window_len):
        test_sample_window = []
        for i in range(train_num, df_rows - predict_days):
            test_sample_window.append(feature_normal[i])
            if len(test_sample_window) == window_len:
                raise_10days = (float(price[i + predict_days] - price[i]
                                      - label_min) / (label_max - label_min))
                yield np.array(test_sample_window), np.array(raise_10days)
                test_sample_window = test_sample_window[1:]
            # 排除跳点
            if (df['UpdateTime'][i + 1] - df['UpdateTime'][i]) > time_delta:
                test_sample_window = []
                # print("jump point")

    def train_gen(): return train_generator(predict_days, window_len)
    def test_gen(): return test_generator(predict_days, window_len)

    output_types = ((tf.float32, tf.float32), (tf.float32))
    output_shapes = ((window_len, feature.shape[1]), (1))
    print('output_shape:', output_shapes)
    train_dataset = tf.data.Dataset.from_generator(
        train_gen, output_types=output_types, output_shapes=output_shapes)
    test_dataset = tf.data.Dataset.from_generator(
        test_gen, output_types=output_types, output_shapes=output_shapes)
    train_dataset = train_dataset.batch(32, drop_remainder=True)
    test_dataset = test_dataset.batch(32, drop_remainder=True)
    return train_dataset, test_dataset


class DataGenerator(keras.utils.Sequence):

    def __init__(self, datafile, predict_days, window_len, batch_size, dataset):
        self.data = pd.read_csv(datafile)
        count, _ = self.data.shape
        print('count:', count)
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

        # 找出跳点
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

        # 总数还要减去跳点造成失效的数据，且知跳点间隔足够大
        self.num = self.df_rows-self.predict_days-self.window_len + \
            1-self.jump_points_num*(self.window_len-1)
        print('jump_points_num:', self.jump_points_num)

        # get label
        # print(self.df.head())
        self.price = self.df['midPrice']
        # print(self.price)
        self.label = []

        for i in range(self.df_rows - self.predict_days):
            self.label.append(
                self.price[self.begin+i + self.predict_days] - self.price[self.begin+i])
        self.label_max = max(self.label)
        self.label_min = min(self.label)
        self.label = [(l-self.label_min)/(self.label_max-self.label_min)
                      for l in self.label]
        # normalize feature
        self.feature = self.df.drop(
            ['midPrice', 'UpdateTime', 'UpdateMillisec'], axis=1)
        self.feature_normal = maxmin_norm(self.feature.values)
        print('num:', self.num)
        print('label:', len(self.label))

    def __len__(self):
        return np.floor(self.num/self.batch_size-1).astype(np.int)

    def get_num(self):
        return self.num

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
                idx += temp * self.window_len
        return prev_jump_num

    def __getitem__(self, idx):
        # print('index', idx)
        batch_x = []
        batch_y = []
        for i in range(idx*self.batch_size, (idx+1)*self.batch_size):
            # 计算窗口因跳点应向后滑动的窗口数
            prev_jump_num = self.get_prev_jump_num(i+self.window_len)
            # 根据此窗口前的跳点数向后滑动窗口
            i += self.window_len*prev_jump_num
            batch_x.append(self.feature_normal[i:i+self.window_len])
            batch_y.append(self.label[i+self.window_len-1])

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        # batch_y = np.array(self.label[idx*self.batch_size:(idx+1)*self.batch_size])

        return batch_x, batch_y

    def get_len(self):
        return np.floor(self.num/self.batch_size).astype(np.int)


class IdentityDataGenerator(keras.utils.Sequence):

    def __init__(self, datafile, batch_size, dataset):
        self.data = pd.read_csv(datafile)
        count, _ = self.data.shape
        print('count:', count)
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
        self.feature_normal = maxmin_norm(self.feature.values)
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


class IdentityDataReader(keras.utils.Sequence):
    def __init__(self, datafile, dataset):
        self.data = pd.read_csv(datafile)
        count, _ = self.data.shape
        print('count:', count)
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
        self.feature_normal = maxmin_norm(self.feature.values)

    def get_data(self):
        return self.feature_normal


if __name__ == "__main__":
    DataGenerator('./dataset/data.csv', 10, 50, 32, 'test')
