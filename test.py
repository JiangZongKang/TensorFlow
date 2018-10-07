#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author: JiangZongKang
@contact: top_jzk@163.com
@file: test.py
@time: 2018/9/4 17:12

"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # 实现一个加法运算
# a = tf.constant(5.0)
# b = tf.constant(6.0)
#
# sum1 = tf.add(a, b)
# plt = tf.placeholder(tf.float32, [2, 3])
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess, tf.device('/gpu:0'):
#     print(sess.run([a, b, sum1]))
#     print(sess.run(plt, feed_dict={plt: [[1, 2, 3], [4, 5, 6]]}))
#
# # print(sum1.eval())


a = tf.constant([1, 2, 3, 4, 5])
var = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0))
var_init = tf.global_variables_initializer()
# print(a,var)
with tf.Session() as sess:
    sess.run(var_init)
    filewriter = tf.summary.FileWriter('./tmp/summary/test', graph=sess.graph)
    print(sess.run([a, var]))
