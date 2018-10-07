#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author: JiangZongKang
@contact: top_jzk@163.com
@file: cifar_tf.py
@time: 2018/9/22 11:00

"""
import tensorflow as tf
import os
import pickle
import numpy as np

# 文件存放目录
cifar_dir = './cifar-10-batches-py'


# print(os.listdir(cifar_dir))

def load_data(filename):
    '''
    read data from data file
    :param filename:
    :return:
    '''
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')  # python3 需要添加上encoding='bytes'
        return data[b'data'], data[b'labels']  # 并且 在 key 前需要加上 b


# tensorflow.Dataset
class CifarData():
    def __init__(self, filenames, need_shuffle):
        '''
        参数1:文件夹
        参数2:是否需要随机打乱
        '''
        all_data = []
        all_labels = []
        for filename in filenames:
            # 将所有的数据,标签分别存放在两个list中
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        # print(all_data)
        # print('-'*100)
        # print(all_labels)
        # 将列表 组成 一个numpy类型的矩阵!!!!
        self._data = np.vstack(all_data)
        # 对数据进行归一化, 尺度固定在 [-1, 1] 之间
        self._data = self._data / 127.5 - 1
        # 将列表,变成一个 numpy 数组
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)
        # 记录当前的样本 数量
        self._num_examples = self._data.shape[0]
        # 保存是否需要随机打乱
        self._need_shuffle = need_shuffle
        # 样本的起始点
        self._indicator = 0
        # 判断是否需要打乱
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # np.random.permutation() 从 0 到 参数,随机打乱
        p = np.random.permutation(self._num_examples)
        # 保存 已经打乱 顺序的数据,其中p是一个一维数组，不是个list，按照行进行打乱
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        '''
        renturn batch_size examples as batch
        :param batch_size: 数据量
        :return: batch_data(样本)  batch_lables(标签)
        '''
        # 开始点 + 数量 = 结束点
        end_indicator = self._indicator + batch_size
        # 如果结束点大于样本数量
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                # 重新打乱
                self._shuffle_data()
                # 开始点归零,从头再来
                self._indicator = 0
                # 重新指定 结束点. 和上面的那一句,说白了就是重新开始
                end_indicator = batch_size  # 其实就是 0 + batch_size, 把 0 省略了
            else:
                raise Exception('hava no more examples')
        # 再次查看是否 超出边界了
        if end_indicator > self._num_examples:
            raise Exception('batch size is larger than all examples')
        # 把 batch 区间 的data和label保存,并最后return
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels

# 拿到所有文件名称
train_filenames = [os.path.join(cifar_dir, 'data_batch_%d' % i) for i in range(1, 6)]
# 拿到标签
test_filenames = [os.path.join(cifar_dir, 'test_batch')]
# 拿到训练数据和测试数据
train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)
# batch_data, batcha_labels = train_data.next_batch(10)
# print(batch_data)
# print(batcha_labels)




# 设计计算图
# 形状 [None, 3072] 3072 是 样本的维数, None 代表位置的样本数量
x = tf.placeholder(tf.float32, [None, 3072])
# 形状 [None] y的数量和x的样本数是对应的
y = tf.placeholder(tf.int64, [None])
x_image = tf.reshape(x,(-1,32,32,3))

# first conv
conv1 = tf.layers.conv2d(x_image,32,(3,3),padding='SAME',activation=tf.nn.relu,name='conv1')
pooling1 = tf.layers.max_pooling2d(conv1,(2,2),(2,2),name='pool1')

# second conv
conv2 = tf.layers.conv2d(pooling1,64,(3,3),padding='same',activation=tf.nn.relu,name='conv2')
pooling2 = tf.layers.max_pooling2d(conv2,(2,2),(2,2),name='pool2')

# third conv
conv3 = tf.layers.conv2d(pooling2,128,(3,3),padding='same',activation=tf.nn.relu,name='conv3')
pooling3 = tf.layers.max_pooling2d(conv3,(2,2),(2,2),name='pool3')

flatten = tf.layers.flatten(pooling3)
y_ = tf.layers.dense(flatten,10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)

predict = tf.argmax(y_,1)
correct_prediction = tf.equal(predict,y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()
batch_size = 50
train_steps = 10000
test_steps = 100

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data,batch_labels = train_data.next_batch(batch_size)
        loss_val,acc_val,_ =sess.run([loss,accuracy,train_op],
                                     feed_dict={x:batch_data,y:batch_labels})
        if(i + 1)% 100 == 0:
            print('[Train] Step: %d, loss: %f, acc: %f' %(i+1,loss_val,acc_val))

        if(i + 1)% 1000 == 0:
            test_data = CifarData(test_filenames,False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run(accuracy,feed_dict = {x:test_batch_data, y:test_batch_labels})
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step: %d, acc: %f'%(i + 1, test_acc))


# 打开文件操作

'''
f = open('./cifar-10-batches-py/data_batch_1','rb')
data = pickle.load(f,encoding='bytes')
x = data[b'data']
y = data[b'labels']
print(x)
print(x.shape)
print('-'*100)
print(y)
print(len(y))

'''
