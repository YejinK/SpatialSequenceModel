import tensorflow as tf
tf.compat.v1.disable_eager_execution()

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
        with tf.compat.v1.name_scope('input-layer'):
            self.input_extra = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, 20], name='input-extra')
            self.input_inflow = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, self.x, self.x, self.t], name='input-inflow')
            self.input_outflow = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, self.x, self.x, self.t], name='input-outflow')
            self.target_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.batch_size, self.x, self.x], name='target-y')

    def prediction_layer(self):
        with tf.compat.v1.name_scope('prediction-layer'):
            self.outflow_out = self.flow_extract_layer(self.input_outflow, name='outflow_extract_out')
            self.extract_out = self.inflow_attention_layer_2()
            self.flatten_extract_out = tf.compat.v1.layers.flatten(self.extract_out, name='flatten_extract')
            self.final_out = self.merge_layer_2(self.flatten_extract_out)

    def loss_layer(self):
        with tf.compat.v1.name_scope('loss-layer'):
            target = tf.compat.v1.layers.flatten(self.target_y, name='flatten_target')
            self.loss = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(self.final_out - target)), name='loss')

    def merge_layer(self, inputs):
        extra_embedding = tf.keras.layers.Dense(64, activation=tf.nn.relu6, name='extra_embedding')(self.input_extra)
        extra_out = tf.keras.layers.Dense(1024, activation=tf.nn.relu6, name='extra_out')(extra_embedding)
        self.merge_out = tf.nn.sigmoid(tf.reduce_sum(input_tensor=tf.stack([inputs, extra_out], axis=1), axis=1))
        return self.merge_out

    def merge_layer_2(self, inputs):
        with tf.compat.v1.name_scope('merge-layer'):
            extra_embedding = tf.keras.layers.Dense(64, activation=tf.nn.relu6)(self.input_extra)
            extra_out = tf.keras.layers.Dense(1024, activation=tf.nn.relu6)(extra_embedding)
            merge_out = tf.reduce_sum(input_tensor=tf.stack([inputs, extra_out], axis=1), axis=1)
            multiply_layer_1 = self.multiply_layer(merge_out, name='multiply_1')
            multiply_layer_2 = self.multiply_layer(multiply_layer_1, name='multiply_2')
            prediction = tf.nn.sigmoid(multiply_layer_2, 'prediction')
            return prediction

    def inflow_attention_layer(self):
        inflow_extract_out = self.flow_extract(self.input_inflow, name='inflow')
        attention_out = tf.subtract(inflow_extract_out, self.outflow_out)
        return attention_out

    def inflow_attention_layer_2(self):
        with tf.compat.v1.name_scope('attention-layer'):
            inflow_extract_out = self.flow_extract(self.input_inflow, name='inflow')
            flatten_inflow = tf.compat.v1.layers.flatten(inflow_extract_out, name='flatten_inflow_extract')
            flatten_outflow = tf.compat.v1.layers.flatten(self.outflow_out, name='flatten_inflow_extract')
            subtract_out = tf.subtract(flatten_inflow, flatten_outflow)
            attention_layer_1 = self.multiply_layer(subtract_out, name='inflow_attention_layer_1')
            attention_layer_2 = self.multiply_layer(attention_layer_1, name='inflow_attention_layer_2')
        return attention_layer_2

    def multiply_layer(self, inputs, name):
        multiply_w = tf.Variable(tf.random.truncated_normal([1024], stddev=1), name=name + '_multiply_w')
        multiply_b = tf.Variable(tf.constant(0.1, shape=[1]), name=name + '_multiply_b')
        return tf.multiply(multiply_w, inputs) + multiply_b

    def fc_layer(self, inputs, activation=None, name=None):
        return tf.keras.layers.Dense(1024, activation=activation, name=name)(inputs)

    ##########################################################################################################
    #                   Feature extraction layer based on the influence of positional relationship
    ##########################################################################################################
    def flow_extract_layer(self, inputs, name):
        with tf.compat.v1.name_scope('flow-extract-layer'):
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
