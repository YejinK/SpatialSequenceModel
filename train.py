import tensorflow as tf
import numpy as np
import time

from model_2 import Model
from data_helper import *


def test(batch, x, t, r):
    model = Model(batch, x, t, r)
    helper = DataHelper()

    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())

        inflow, outflow, target, attention, extra, series = helper.generate(batch, t, r)

        out = session.run([model.extract_out], feed_dict={model.input_inflow: inflow,
                                                          model.input_outflow: outflow,
                                                          model.target_y: target,
                                                          model.input_extra: extra})
        out = np.array(out[0])
        # show(out[0])
        print(out.shape)


def train(batch_size, x, t, r, lr, lr_step, lr_rate, steps):
    model = Model(batch_size, x, t, r)
    helper = DataHelper()

    final_RMSE = 0

    global_step = tf.Variable(0, trainable=False)
    dynamic_lr = tf.compat.v1.train.exponential_decay(lr, global_step, lr_step, lr_rate, staircase=True)

    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=dynamic_lr).minimize(model.loss, global_step=global_step)

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=3)
    save_timestamp = str(int(time.time()))

    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())

        start_time = time.time()

        for step in range(1, steps + 1):
            inflow, outflow, target, attention, extra, series = helper.generate(batch_size, t, r)

            _, RMSE, dlr, = session.run([train_step, model.loss, dynamic_lr], feed_dict={model.input_inflow: inflow,
                                                                                         model.input_outflow: outflow,
                                                                                         model.target_y: target,
                                                                                         model.input_extra: extra})

            if step % 100 == 0:
                end_time = time.time()
                print('{} steps in {:.4}s. lr={:.6}, RMSE={:.6}, RMSE(real)={:.6}.'
                      .format(step, end_time - start_time, dlr, RMSE, helper.reverse(RMSE)))

                start_time = end_time

            if step % 100 == 0:
                inflow, outflow, series, extra, target = helper.eval_data(batch, t, r)
                eval_RMSE = session.run(model.loss, feed_dict={model.input_inflow: inflow,
                                                               model.input_outflow: outflow,
                                                               model.target_y: target,
                                                               model.input_extra: extra})

                final_RMSE = eval_RMSE
                RMSE_real = helper.reverse(eval_RMSE)
                print('With {} steps. Evaluation RMSE(real)={:.6}'.format(step, RMSE_real))

            if step % 1000 == 0:
                current_step = tf.compat.v1.train.global_step(session, global_step)
                # TODO: path change
                saver.save(session, "/home/ykim/workspace/SpatialSequenceModel/model/" + save_timestamp + "/model", global_step=current_step)

        record_model(batch_size, x, t, lr, lr_step, lr_rate, steps, final_RMSE, helper.reverse(final_RMSE))


def parameter_settings():
    # 基础参数
    batch_size = 8
    x = 32
    t = 4
    r = 1

    # 学习率调节
    lr = 0.0018
    lr_step = 50
    lr_rate = 0.99

    # 循环参数
    steps = 10000

    return batch_size, x, t, r, lr, lr_step, lr_rate, steps


if __name__ == '__main__':
    batch, x, t, r, lr, lr_step, lr_rate, steps = parameter_settings()
    train(batch, x, t, r, lr, lr_step, lr_rate, steps)
    # test(batch, x, t, r)
