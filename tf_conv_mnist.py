#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author: JiangZongKang
@contact: top_jzk@163.com
@file: tf_conv_mnist.py
@time: 2018/9/15 10:24

"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 定义一个初始化权重的函数
def weigtht_variables(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


# 定义一个初始化偏置的函数
def bias_variables(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def model():
    '''
    自定义的卷积模型
    :return:
    '''
    # 1. 准备数据的占位符 x [None,784] y_true [None,10]
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32, [None, 784])
        # print(x)
        y_true = tf.placeholder(tf.int32, [None, 10])

    # 2. 一卷积层 卷积、激活、池化
    with tf.variable_scope('conv1'):
        # 随机初始化权重
        w_conv1 = weigtht_variables([5, 5, 1, 32])
        # 初始化偏置
        b_conv1 = bias_variables([32])
        # 对x的形状进行改变 [None,784]------>[None,28,28,1]
        x_shape = tf.reshape(x, [-1, 28, 28, 1])
        print(x_shape.shape)
        # [None,28,28,1]-------->[None,28,28,32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_shape, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        # 池化
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 3. 二卷积层
    with tf.variable_scope('conv2'):
        # 随机初始化权重
        w_conv2 = weigtht_variables([5, 5, 32, 64])
        # 初始化偏置
        b_conv2 = bias_variables([64])
        # [None,14,14,32]-------->[None，14,14,64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        # 池化 [None,14,14,64]------->[None,7,7,64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 4. 全连接层
    with tf.variable_scope('fc'):
        # 随机初始化权重
        w_fc = weigtht_variables([7 * 7 * 64, 10])
        # 初始化偏置
        b_fc = bias_variables([10])
        # 修改形状 [None,7,7,64]-------->[None,7*7*64]
        x_fc = tf.reshape(x_pool2, [-1, 7 * 7 * 64])
        # 进行矩阵运算得出10个样本的结果
        y_pred = tf.matmul(x_fc, w_fc) + b_fc
    return x, y_true, y_pred


def conv_fc():
    '''
    训练模型
    :return:
    '''
    # 获取真实的数据
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    tf.logging.set_verbosity(old_v)

    # 定义模型，得出输出
    x, y_true, y_pred = model()

    # 进行交叉熵损失计算
    with tf.variable_scope('soft_cross'):
        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_true, logits=y_pred))

    # 梯度下降求出损失
    with tf.variable_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    # 计算准确率
    with tf.variable_scope('acc'):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 开启会话去训练
    with tf.Session() as sess:
        sess.run(init_op)
        # 循环去训练
        for i in range(20000):
            # 取出真实存在的目标值和特征值
            mnist_x, mnist_y = mnist.train.next_batch(50)
            # 运行train_op 训练
            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

            if (i+1) % 100 == 0:
                print('训练第%d步，准确率为：%f' % (i+1,
                                      sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))
        # sess.run(accuracy,feed_dict={x: mnist.test.images, y_true: mnist.test.labels})

        # 在测试集上进行测试
        print('最后在测试集上的准确为：%f' %(sess.run(accuracy,
                                          feed_dict={x: mnist.test.images[:2000],
                                                     y_true: mnist.test.labels[:2000]})))


    return None


if __name__ == '__main__':
    conv_fc()
