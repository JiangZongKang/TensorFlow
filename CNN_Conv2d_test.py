#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author: JiangZongKang
@contact: top_jzk@163.com
@file: CNN_Conv2d_test.py.py
@time: 2018/9/14 14:37

"""
import numpy as np
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense,Input

# -----------通用模型---------------
# 定义输入层
input = Input(shape=(784,))
# 定义各个连接层
x = Dense(64,activation='relu')(input)
x = Dense(64,activation='relu')(input)
# 定义输出层
y = Dense(10,activation='softmax')(x)

# 定义模型
model = Model(inputs=input,outputs=y)


