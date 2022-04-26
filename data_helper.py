import numpy as np
import h5py

import random
import time


class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

class MinMaxNormalization_01(object):
    """
    MinMax Normalization --> [0, 1]
    x = (x - min) / (max - min).
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        # print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = 1. * X * (self._max - self._min) + self._min
        return X


class DataHelper(object):
    def __init__(self):
        self.inflow_mmn = MinMaxNormalization_01()
        self.outflow_mmn = MinMaxNormalization_01()

        # TODO: path change
        self.raw_data = h5py.File('/home/ykim/workspace/SpatialSequenceModel/data/BJ13_M32x32_T30_InOut.h5')
        self.extra_data = h5py.File('/home/ykim/workspace/SpatialSequenceModel/data/BJ_Meteorology.h5')

        self.inflow_mmn.fit(self.raw_data['data'][:][0])
        self.outflow_mmn.fit(self.raw_data['data'][:][1])

    def generate(self, batch_size, t, r):
        in_data = []
        out_data = []
        label = []
        extra = []
        attention = []
        series_data = []

        for _ in range(batch_size):
            random_time = random.randint(t, 2000)

            # inflow data
            in_arr = []
            for j in range(t):
                in_arr.append(self.inflow_mmn.transform(self.raw_data['data'][random_time - j][0]))
            in_arr = np.swapaxes(in_arr, axis1=0, axis2=2)
            in_arr = np.swapaxes(in_arr, axis1=0, axis2=1)
            in_data.append(in_arr)

            # outflow data
            out_arr = []
            for j in range(t):
                out_arr.append(self.outflow_mmn.transform(self.raw_data['data'][random_time - j][1]))
            out_arr = np.swapaxes(out_arr, axis1=0, axis2=2)
            out_arr = np.swapaxes(out_arr, axis1=0, axis2=1)
            out_data.append(out_arr)

            # series data
            series_arr = []
            for j in range(r):
                series_arr.append(self.inflow_mmn.transform(self.raw_data['data'][random_time - j][0]))
            series_arr = np.swapaxes(series_arr, axis1=0, axis2=2)
            series_arr = np.swapaxes(series_arr, axis1=0, axis2=1)
            series_data.append(series_arr)

            # label data
            label.append(self.inflow_mmn.transform(self.raw_data['data'][random_time + 1][0]))

            # attention data
            attention_data_inflow = [self.inflow_mmn.transform(self.raw_data['data'][random_time][0])]
            attention_data_inflow = np.swapaxes(attention_data_inflow, axis1=0, axis2=2)
            attention_data_inflow = np.swapaxes(attention_data_inflow, axis1=0, axis2=1)
            attention_data_outflow = [self.inflow_mmn.transform(self.raw_data['data'][random_time][1])]
            attention_data_outflow = np.swapaxes(attention_data_outflow, axis1=0, axis2=2)
            attention_data_outflow = np.swapaxes(attention_data_outflow, axis1=0, axis2=1)
            attention.append(attention_data_inflow)
            attention.append(attention_data_outflow)

            # extra data
            extra_list = list(map(float, self.extra_data['Weather'][random_time + 1]))
            extra_list.insert(0, self.extra_data['Temperature'][random_time + 1])
            extra_list.insert(0, self.extra_data['WindSpeed'][random_time + 1])
            extra_list.insert(0, self.extra_data['date'][random_time + 1][-2:])
            extra.append(extra_list)

        return np.array(in_data), \
               np.array(out_data), \
               np.array(label), \
               np.reshape(attention, [-1, 32 * 32 * 2]), \
               np.array(extra), \
               np.array(series_data)

    def eval_data(self, batch_size, t, r):
        in_data = []
        out_data = []
        label = []
        extra = []
        series_data = []

        for _ in range(batch_size):
            random_time = random.randint(2000, 4800)

            # inflow data
            in_arr = []
            for j in range(t):
                in_arr.append(self.inflow_mmn.transform(self.raw_data['data'][random_time - j][0]))
            in_arr = np.swapaxes(in_arr, axis1=0, axis2=2)
            in_arr = np.swapaxes(in_arr, axis1=0, axis2=1)
            in_data.append(in_arr)

            # outflow data
            out_arr = []
            for j in range(t):
                out_arr.append(self.outflow_mmn.transform(self.raw_data['data'][random_time - j][1]))
            out_arr = np.swapaxes(out_arr, axis1=0, axis2=2)
            out_arr = np.swapaxes(out_arr, axis1=0, axis2=1)
            out_data.append(out_arr)

            # series data
            series_arr = []
            for j in range(r):
                series_arr.append(self.inflow_mmn.transform(self.raw_data['data'][random_time - j][0]))
            series_arr = np.swapaxes(series_arr, axis1=0, axis2=2)
            series_arr = np.swapaxes(series_arr, axis1=0, axis2=1)
            series_data.append(series_arr)

            # label data
            label.append(self.inflow_mmn.transform(self.raw_data['data'][random_time + 1][0]))

            # extra data
            extra_list = list(map(float, self.extra_data['Weather'][random_time + 1]))
            extra_list.insert(0, self.extra_data['Temperature'][random_time + 1])
            extra_list.insert(0, self.extra_data['WindSpeed'][random_time + 1])
            extra_list.insert(0, self.extra_data['date'][random_time + 1][-2:])
            extra.append(extra_list)

        return np.array(in_data), \
               np.array(out_data), \
               np.array(series_data), \
               np.array(extra), \
               np.array(label)

    def eval_data_all(self, t, r):
        inflow_list = []
        outflow_list = []
        series_list = []
        extra_list = []
        label_list = []

        for i in range(3000, 4800):
            inflow, outflow, series, extra, label = self.generate_data(i, t, r)
            inflow_list.append(inflow)
            outflow_list.append(outflow)
            series_list.append(series)
            extra_list.append(extra)
            label_list.append(label)

        return np.array(inflow_list), np.array(outflow_list), np.array(series_list), np.array(extra_list), np.array(label_list)

    def generate_data(self, time, t, r):
        # inflow data
        inflow_data = []
        for j in range(t):
            inflow_data.append(self.inflow_mmn.transform(self.raw_data['data'][time - j][0]))
        inflow_data = np.swapaxes(inflow_data, axis1=0, axis2=2)
        inflow_data = np.swapaxes(inflow_data, axis1=0, axis2=1)

        # outflow data
        outflow_data = []
        for j in range(t):
            outflow_data.append(self.outflow_mmn.transform(self.raw_data['data'][time - j][1]))
        outflow_data = np.swapaxes(outflow_data, axis1=0, axis2=2)
        outflow_data = np.swapaxes(outflow_data, axis1=0, axis2=1)

        # series data
        series_data = []
        for j in range(r):
            series_data.append(self.inflow_mmn.transform(self.raw_data['data'][time - j][0]))
        series_data = np.swapaxes(series_data, axis1=0, axis2=2)
        series_data = np.swapaxes(series_data, axis1=0, axis2=1)

        # label data
        label_data = self.inflow_mmn.transform(self.raw_data['data'][time + 1][0])

        # extra data
        extra_data = list(map(float, self.extra_data['Weather'][time + 1]))
        extra_data.insert(0, self.extra_data['Temperature'][time + 1])
        extra_data.insert(0, self.extra_data['WindSpeed'][time + 1])
        extra_data.insert(0, self.extra_data['date'][time + 1][-2:])

        return inflow_data, outflow_data, series_data, extra_data, label_data

    def reverse(self, X):
        return X * (self.inflow_mmn._max - self.inflow_mmn._min) / 2.


def show(matrix):
    for line in matrix:
        for value in line:
            print('{:.4f}'.format(value[0]), end='\t')
        print('\n')


def record_model(batch, x, t, lr, decay_steps, decay_rate, steps, rmse, real):
    # TODO: path change
    f = open('/home/ykim/workspace/SpatialSequenceModel/model/RECORD', mode='a+')
    format_data = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    timestamp = str(int(time.time()))
    model = 'No.{} Model, Training on {}'.format(timestamp, format_data)
    content = 'batch={},x={},t={},steps={}'.format(batch, x, t, steps)
    learning_rate = 'lr={:.6f},decay_steps={},decay_rate={:.6f}'.format(lr, decay_steps, decay_rate)
    result = 'RMSE={:.6f},RMSE(real)={:.6f}'.format(rmse, real)
    f.writelines(model + '\n')
    f.writelines(content + '\n')
    f.writelines(learning_rate + '\n')
    f.writelines(result + '\n')
    f.writelines('\n')


if __name__ == "__main__":
    # print(generate_data(8, 5))
    # generate_map_data(2, 4)
    helper = DataHelper()
    inflow, outflow, series, extra, label = helper.eval_data_all(4, 3)
    # print(data)
    # print(label)
    print(series.shape)
    # print(helper.reverse(label))
    # extra_data = h5py.File('BJ_Meteorology.h5')
    # print(extra_data['weather'])
    # for key in extra_data.keys():
    #     print(key)
