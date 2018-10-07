#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author: JiangZongKang
@contact: top_jzk@163.com
@file: mnist_test.py
@time: 2018/9/23 20:30

"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# def test_mnist():
#     '''
#     use InteractiveSession environment to train and test mnist dataset
#     :return:
#     '''
#     mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#     sess = tf.InteractiveSession()
#     x = tf.placeholder(tf.float32, [None, 784])
#     y_ = tf.placeholder(tf.float32, [None, 10])
#
#     w = tf.Variable(tf.random_normal([784, 10]))
#     b = tf.Variable(tf.constant(0.0, shape=[10]))
#
#     sess.run(tf.initialize_all_variables())
#
#     y = tf.nn.softmax(tf.matmul(x, w) + b)
#     cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#
#     train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#
#     for i in range(2000):
#         batch = mnist.train.next_batch(50)
#         train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#
#     correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
#     print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#
#     return None
#
#
# if __name__ == '__main__':
#     test_mnist()
