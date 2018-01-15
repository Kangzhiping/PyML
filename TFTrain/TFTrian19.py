# 时间序列数据
# 时间序列数据是指在不同时间点上收集到的数据，这类数据反映了某一事物、现象等随时间的变化状态或程度。

# 简单预测一下未来几个月天朝铁路客运量

# 画出走势
import matplotlib.pyplot as plt
import pandas as pd
#import requests
#import io
import numpy as np

#url = 'http://'
#ass_data = requests.get(url).content

#df = pd.read_csv(io.StringIO(ass_data.decode('utf-8')))  # python2使用StringIO.StringIO
fileIn=open('data/铁路客运量.csv',encoding='utf-8')
df = pd.read_csv(fileIn)

data = np.array(df['铁路客运量_当期值(万人)'])
# normalize
normalized_data = (data - np.mean(data)) / np.std(data)

plt.figure()
plt.plot(data)
plt.show()

# 构建网络做预测

import tensorflow as tf

seq_size = 3
train_x, train_y = [], []
for i in range(len(normalized_data) - seq_size - 1):
    train_x.append(np.expand_dims(normalized_data[i: i + seq_size], axis=1).tolist())
    train_y.append(normalized_data[i + 1: i + seq_size + 1].tolist())

input_dim = 1
X = tf.placeholder(tf.float32, [None, seq_size, input_dim])
Y = tf.placeholder(tf.float32, [None, seq_size])


# regression
def ass_rnn(hidden_layer_size=6):
    W = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W')
    b = tf.Variable(tf.random_normal([1]), name='b')
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    W_repeated = tf.tile(tf.expand_dims(W, 0), [tf.shape(X)[0], 1, 1])
    out = tf.batch_matmul(outputs, W_repeated) + b
    out = tf.squeeze(out)
    return out


def train_rnn():
    out = ass_rnn()

    loss = tf.reduce_mean(tf.square(out - Y))
    train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # tf.get_variable_scope().reuse_variables()
        sess.run(tf.global_variables_initializer())

        for step in range(10000):
            _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})
            if step % 10 == 0:
                # 用测试数据评估loss
                print(step, loss_)
        print("保存模型: ", saver.save(sess, 'ass.model'))


# train_rnn()

def prediction():
    out = ass_rnn()

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # tf.get_variable_scope().reuse_variables()
        saver.restore(sess, './ass.model')

        prev_seq = train_x[-1]
        predict = []
        for i in range(12):
            next_seq = sess.run(out, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        plt.figure()
        plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
        plt.plot(list(range(len(normalized_data), len(normalized_data) + len(predict))), predict, color='r')
        plt.show()

        # prediction()

#红色曲线是预测值
# 使用TensorBoard绘出loss和准确率 来判断已经训练到最优。
