from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
import urllib
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow as tf

mnist = read_data_sets("MNIST_data/", one_hot=True)
#print(mnist)

'''x不是一个特定的值，而是一个占位符placeholder，我们在TensorFlow运行计算时输入这个值。我们希望能够输入
任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。
（这里的None表示此张量的第一个维度可以是任何长度的。）'''
x = tf.placeholder(tf.float32,[None, 784])

'''我们的模型也需要权重值和偏置量，当然我们可以把它们当做是另外的输入（使用占位符），
但TensorFlow有一个更好的方法来表示它们：Variable 。 一个Variable代表一个可修改的张量，
存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。
对于各种机器学习应用，一般都会有模型参数，可以用Variable表示。'''

w = tf.Variable(tf.zeros([784,10]))
b= tf.Variable(tf.zeros([10]))

#实现
#softmax模型可以用来给不同的对象分配概率。即使在之后，我们训练更加精细的模型时，最后一步也需要用softmax来分配概率
'''softmax可以看成是一个激励（activation）函数或者链接（link）函数，把我们定义的线性函数的输出转换成我们
想要的格式，也就是关于10个数字类的概率分布。因此，给定一张图片，它对于每一个数字的吻合度可以被softmax函数
转换成为一个概率值'''
y = tf.nn.softmax(tf.matmul(x,w) + b)

'''定义一个指标来评估这个模型是好的。其实，在机器学习，我们通常定义指标来表示一个模型是坏的，
这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标'''
#成本函数是“交叉熵”（cross-entropy）
#y 是我们预测的概率分布, y' 是实际的分布（我们输入的one-hot vector)
y_ = tf.placeholder("float",[None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

'''TensorFlow拥有一张描述你各个计算单元的图，它可以自动地使用反向传播算法(backpropagation algorithm)
来有效地确定你的变量是如何影响你想要最小化的那个成本值的。然后，TensorFlow会用你选择的优化算法来不断地
修改变量以降低成本.
要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。梯度下降算法
（gradient descent algorithm）是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的
方向移动。当然TensorFlow也提供了其他许多优化算法：只要简单地调整一行代码就可以使用其他的算法。
TensorFlow在这里实际上所做的是，它会在后台给描述你的计算的那张图里面增加一系列新的计算操作单元用于实现反
向传播算法和梯度下降算法。然后，它返回给你的只是一个单一的操作，当运行这个操作时，它用梯度下降算法训练你
的模型，微调你的变量，不断减少成本'''

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    #评估我们的模型
    '''首先让我们找出那些预测正确的标签。tf.argmax 是一个非常有用的函数，它能给出某个tensor对象
    在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，因此最大值1所在的索引
    位置就是类别标签，比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而
    tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹
    配(索引位置一样表示匹配)。'''

    correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))

    '''这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，
    然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.'''

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 我们计算所学习到的模型在测试数据集上面的正确率
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html