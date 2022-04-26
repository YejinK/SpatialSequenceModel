import tensorflow as tf
from data_helper import *
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


class Evaluation:

    def __init__(self, folder, size):
        self.folder = folder
        self.steps = size

    def eval_model(self, t, r):
        checkpoint_file = tf.train.latest_checkpoint("model/{}/".format(self.folder))
        graph = tf.Graph()
        with graph.as_default():
            self.session = tf.compat.v1.Session()
            with self.session.as_default():
                # 加载 .meta 图与变量
                saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.session, checkpoint_file)
                self.input_inflow = graph.get_operation_by_name("input-layer/input-inflow").outputs[0]
                self.input_outflow = graph.get_operation_by_name("input-layer/input-outflow").outputs[0]
                self.target_y = graph.get_operation_by_name("input-layer/target-y").outputs[0]
                self.input_extra = graph.get_operation_by_name("input-layer/input-extra").outputs[0]
                self.prediction = graph.get_operation_by_name("prediction-layer/merge-layer/prediction").outputs[0]
                self.loss = graph.get_operation_by_name("loss-layer/loss").outputs[0]

                self.eval_data(self.steps, 8, t, r)
                # self.eval_visualized(8, t, r)

    def eval_data(self, steps, batch, t, r):

        helper = DataHelper()

        predictions = []
        loss_list = []
        target = []

        for i in range(1, steps + 1):
            inflow, outflow, series, extra, target = helper.eval_data(batch, t, r)
            loss, prediction = self.session.run([self.loss, self.prediction], feed_dict={self.input_inflow: inflow,
                                                                                         self.input_outflow: outflow,
                                                                                         self.target_y: target,
                                                                                         self.input_extra: extra})
            # predictions.append(prediction)
            # target.append(label)
            loss_list.append(loss)

            if (i % 5 == 0):
                RMSE = np.mean(loss_list)
                print("{} data have been validated. RMSE={:.6}, RMSE(real)={:.6}.".format(i * batch, RMSE,
                                                                                          helper.reverse(RMSE)))

    def eval_visualized(self, batch, t, r):
        helper = DataHelper()

        inflow, outflow, series, extra, target = helper.eval_data(batch, t, r)
        loss, prediction = self.session.run([self.loss, self.prediction], feed_dict={self.input_inflow: inflow,
                                                                                     self.input_outflow: outflow,
                                                                                     self.target_y: target,
                                                                                     self.input_extra: extra})

        p = helper.reverse(np.reshape(prediction[-1], [32, 32]))
        t = helper.reverse(np.reshape(target[-1], [32, 32]))
        l = np.abs(t - p)

        fig = plt.figure(figsize=(11, 3))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        sns.heatmap(p, annot=False, vmax=200, vmin=10, square=True, cmap='RdYlGn_r', ax=ax1)
        sns.heatmap(t, annot=False, vmax=200, vmin=10, square=True, cmap='RdYlGn_r', ax=ax2)
        sns.heatmap(l, annot=False, vmax=50, vmin=10, square=True, cmap='Reds', ax=ax3)

        plt.show()


if __name__ == "__main__":
    evaluation = Evaluation("1556178571", size=10000)
    evaluation.eval_model(t=4, r=1)
