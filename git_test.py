#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author: JiangZongKang
@contact: top_jzk@163.com
@file: git_test.py.py
@time: 2018/10/1 19:23

"""


import tensorflow as tf
import os
import pickle
import numpy as np

cifar_dir = './cifar-10-batches-py'

def load_data(filename):
    '''
    read data from data file
    :param filename:
    :return:
    '''
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


class CifarData():
    def __init__(self, filenames):
        # all_data = []
        # all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            # all_data.append(data)
            # all_labels.append(labels)
        # print(all_data)
        # print('-'*100)
        # print(all_labels)
        # self._data = np.vstack(all_data)
        # self._data = self._data / 127.5 - 1
        # self._labels = np.hstack(all_labels)
        # print(self._data.shape)
        # print(self._labels.shape)

        # self._num_examples = self._data.shape[0]
        # self._need_shuffle = need_shuffle
        # self._indicator = 0
        # if self._need_shuffle:
        #     self._shuffle_data()

train_filenames = [os.path.join(cifar_dir, 'data_batch_%d' % i) for i in range(1,6)]
print(train_filenames)
train_data = CifarData(train_filenames)

f = open('./cifar-10-batches-py/data_batch_1','rb')
data = pickle.load(f,encoding='bytes')
x = data[b'data']
y = data[b'labels']
print(x)
print(type(x))
print(x.shape)
print('-'*100)
print(y)
print(len(y))







