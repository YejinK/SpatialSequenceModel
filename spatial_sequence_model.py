import tensorflow as tf
import keras
import numpy as np
import time
from data_helper import *


class Model:
    def __init__(self, batch_size, x, t, r):
        self.batch_size = batch_size
        self.x = x
        self.t = t
        self.r = r

        self.input_layer()
        self.prediction_layer()
        self.loss_layer()

    def input_layer(self):
        self.input_extra = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, 20], name='input-extra')
        self.input_inflow = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, self.x, self.x, self.t], name='input-inflow')
        self.input_outflow = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, self.x, self.x, self.t], name='input-outflow')
        self.input_series = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, self.x, self.x, self.r], name='input-series')
        self.target_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, self.x, self.x], name='target-y')

    def prediction_with_lstm_layer(self):
        self.inflow_out = self.flow_extract_layer(self.input_inflow, name='inflow_extract_out')
        self.outflow_out = self.outflow_attention_layer(self.input_outflow)
        self.extract_out = tf.subtract(self.inflow_out, self.outflow_out, name='prediction')
        self.series_out = self.series_layer()
        self.prediction = self.merge_layer(self.series_out)
        return self.prediction

    def prediction_layer(self):
        self.inflow_out = self.flow_extract_layer(self.input_inflow, name='inflow_extract_out')
        self.outflow_out = self.outflow_attention_layer(self.input_outflow)
        self.extract_out = tf.subtract(self.inflow_out, self.outflow_out, name='prediction')
        self.flatten_extract_out = tf.compat.v1.layers.flatten(self.extract_out, name='flatten_extract')
        self.prediction = self.merge_layer(self.flatten_extract_out)
        return self.prediction

    def loss_layer(self):
        target = tf.compat.v1.layers.flatten(self.target_y, name='flatten_target')
        self.loss = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(self.prediction - target)), name='loss')
        return self.loss

    def series_layer(self):
        inflow_series = tf.concat([self.extract_out, self.input_series], axis=3)
        inflow_series = tf.transpose(a=inflow_series, perm=[0, 3, 1, 2])
        inflow_series = tf.reshape(inflow_series, shape=[-1, self.r + 1, 1024])
        self.lstm_out = tf.keras.layers.SimpleRNN(1024, activation=tf.nn.sigmoid, stateful=False)(inflow_series)
        return self.lstm_out

    def merge_layer(self, inputs):
        extra_embedding = tf.keras.layers.Dense(64, activation=tf.nn.relu6)(self.input_extra)
        extra_out = tf.keras.layers.Dense(1024, activation=tf.nn.relu6)(extra_embedding)
        self.merge_out = tf.nn.sigmoid(tf.reduce_sum(input_tensor=tf.stack([inputs, extra_out], axis=1), axis=1))
        return self.merge_out

    def outflow_attention_layer(self, inputs):
        return self.flow_extract(inputs, name='outflow')

    ############################################################################################
    #                                  基于位置关系影响的特征提取层
    ############################################################################################
    def flow_extract_layer(self, inputs, name):
        left_out = self.flow_extract_left_layer(inputs, name=name)
        left_top_out = self.flow_extract_left_top_layer(inputs, name=name)
        top_out = self.flow_extract_top_layer(inputs, name=name)
        right_top_out = self.flow_extract_right_top_layer(inputs, name=name)
        right_out = self.flow_extract_right_layer(inputs, name=name)
        right_bottom_out = self.flow_extract_right_bottom_layer(inputs, name=name)
        bottom_out = self.flow_extract_bottom_layer(inputs, name=name)
        left_bottom_out = self.flow_extract_left_bottom_layer(inputs, name=name)
        extract_list = [left_out, left_top_out, top_out, right_top_out, right_out, right_bottom_out, bottom_out, left_bottom_out]
        return tf.reduce_sum(input_tensor=tf.concat(extract_list, axis=3), axis=3, keepdims=True, name=name)

    def flow_extract_left_layer(self, inputs, name):
        conv_out = self.flow_extract(inputs, name=name + '_left')
        padding = tf.zeros(shape=[self.batch_size, self.x, 1, 1])
        return tf.concat([padding, conv_out], axis=2)[:, :, :32, :]

    def flow_extract_left_top_layer(self, inputs, name):
        conv_out = self.flow_extract(inputs, name=name + '_left_top')
        top_bottom_padding = tf.zeros(shape=[self.batch_size, 1, self.x, 1])
        padding_1 = tf.concat([top_bottom_padding, conv_out, top_bottom_padding], axis=1)
        left_right_padding = tf.zeros(shape=[self.batch_size, self.x + 2, 1, 1])
        padding_2 = tf.concat([left_right_padding, padding_1, left_right_padding], axis=2)
        return padding_2[:, :32, :32, :]

    def flow_extract_top_layer(self, inputs, name):
        conv_out = self.flow_extract(inputs, name=name + '_top')
        padding = tf.zeros(shape=[self.batch_size, 1, self.x, 1])
        return tf.concat([padding, conv_out], axis=1)[:, :32, :, :]

    def flow_extract_right_top_layer(self, inputs, name):
        conv_out = self.flow_extract(inputs, name=name + '_right_top')
        top_bottom_padding = tf.zeros(shape=[self.batch_size, 1, self.x, 1])
        padding_1 = tf.concat([top_bottom_padding, conv_out, top_bottom_padding], axis=1)
        left_right_padding = tf.zeros(shape=[self.batch_size, self.x + 2, 1, 1])
        padding_2 = tf.concat([left_right_padding, padding_1, left_right_padding], axis=2)
        return padding_2[:, :32, 2:, :]

    def flow_extract_right_layer(self, inputs, name):
        conv_out = self.flow_extract(inputs, name=name + '_right')
        padding = tf.zeros(shape=[self.batch_size, self.x, 1, 1])
        return tf.concat([conv_out, padding], axis=2)[:, :, 1:, :]

    def flow_extract_right_bottom_layer(self, inputs, name):
        conv_out = self.flow_extract(inputs, name=name + '_right_top')
        top_bottom_padding = tf.zeros(shape=[self.batch_size, 1, self.x, 1])
        padding_1 = tf.concat([top_bottom_padding, conv_out, top_bottom_padding], axis=1)
        left_right_padding = tf.zeros(shape=[self.batch_size, self.x + 2, 1, 1])
        padding_2 = tf.concat([left_right_padding, padding_1, left_right_padding], axis=2)
        return padding_2[:, 2:, 2:, :]

    def flow_extract_bottom_layer(self, inputs, name):
        conv_out = self.flow_extract(inputs, name=name + '_bottom')
        padding = tf.zeros(shape=[self.batch_size, 1, self.x, 1])
        return tf.concat([conv_out, padding], axis=1)[:, 1:, :, :]

    def flow_extract_left_bottom_layer(self, inputs, name):
        conv_out = self.flow_extract(inputs, name=name + '_right_top')
        top_bottom_padding = tf.zeros(shape=[self.batch_size, 1, self.x, 1])
        padding_1 = tf.concat([top_bottom_padding, conv_out, top_bottom_padding], axis=1)
        left_right_padding = tf.zeros(shape=[self.batch_size, self.x + 2, 1, 1])
        padding_2 = tf.concat([left_right_padding, padding_1, left_right_padding], axis=2)
        return padding_2[:, 2:, :32, :]

    def flow_extract(self, inputs, out_channel=1, name=None):
        return tf.compat.v1.layers.Conv2D(filters=out_channel, kernel_size=[1, 1], activation=tf.nn.relu6, name=name + '_extract')(inputs)
