#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author: JiangZongKang
@contact: top_jzk@163.com
@file: vgg_train.py
@time: 2018/9/14 19:59

"""

import tensorflow as tf
import numpy as np

x = np.array([i for i in range(1,33)]).reshape((2,2,2,4)) # 转换为一个4维数组，相乘得32
print(x)
y = tf.nn.lrn(input=x,depth_radius=2,bias=0,alpha=1,beta=1)

with tf.Session() as sess:
    # sess.run(y.eval())
    print(x)
    print('------------------------------------------------------------')
    print(y.eval())


