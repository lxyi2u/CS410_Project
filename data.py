import pandas as pd
import numpy as np
import h5py
import os
import tensorflow as tf


class HDF5DatasetWriter:
    def __init__(self, data_dims, labels_dims, outputPath, bufSize=20):
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath' already"
                             "exists and cannot be overwritten. Manually delete"
                             "the file before continuing", outputPath)

        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset("data", data_dims, maxshape=(None,) + data_dims[1:], dtype="float")
        self.labels = self.db.create_dataset("labels", labels_dims, maxshape=(None,) + labels_dims[1:], dtype="float")
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
    maxcols = array.max(axis = 0)
    mincols = array.min(axis = 0)
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

    trainWriter = HDF5DatasetWriter((100, window_len, feature.shape[1]), (100,), './train')
    testWrite = HDF5DatasetWriter((100, window_len, feature.shape[1]), (100,), './test')

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
    sample_window = []
    for i in range(df_rows - predict_days):
        sample_window.append(feature_normal[i])
        if len(sample_window) == window_len:
            raise_10days = (float(price[i + predict_days] - price[i] - label_min) / (label_max - label_min))
            if i <= train_num:
                trainWriter.add(sample_window, raise_10days)
            else:
                testWrite.add(sample_window, raise_10days)
            sample_window = sample_window[1:]
        # 排除跳点
        if (df['UpdateTime'][i + 1] - df['UpdateTime'][i]) > time_delta:
            sample_window = []
            print("跳点")


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
    time_delta = pd.to_datetime('01:00:00', format='%H:%M:%S') - pd.to_datetime('00:00:00', format='%H:%M:%S')


    def train_generator(predict_days, window_len):
        train_sample_window = []
        for i in range (train_num):
            train_sample_window.append(feature_normal[i])
            if len(train_sample_window) == window_len:
                raise_10days = (float(price[i + predict_days] - price[i] - label_min) / (label_max - label_min))
                yield train_sample_window, raise_10days
                train_sample_window = train_sample_window[1:]
            # 排除跳点
            if (df['UpdateTime'][i + 1] - df['UpdateTime'][i]) > time_delta:
                train_sample_window = []
                print("jump point")

    def test_generator(predict_days, window_len):
        test_sample_window = []
        for i in range (train_num, df_rows - predict_days):
            test_sample_window.append(feature_normal[i])
            if len(test_sample_window) == window_len:
                raise_10days = (float(price[i + predict_days] - price[i] - label_min) / (label_max - label_min))
                yield test_sample_window, raise_10days
                test_sample_window = test_sample_window[1:]
            # 排除跳点
            if (df['UpdateTime'][i + 1] - df['UpdateTime'][i]) > time_delta:
                test_sample_window = []
                print("jump point")

    train_gen = lambda: train_generator(predict_days, window_len)
    test_gen = lambda: test_generator(predict_days, window_len)

    output_types = (tf.float32, tf.float32)
    output_shapes = ((window_len, feature.shape[1]), ())

    return (
        tf.data.Dataset.from_generator(train_gen, output_types=output_types, output_shapes=output_shapes),
        tf.data.Dataset.from_generator(test_gen, output_types=output_types, output_shapes=output_shapes)
    )


if __name__ == "__main__":
    DataH5Writer('data.csv', 10, 50)