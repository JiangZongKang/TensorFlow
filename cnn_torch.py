#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author: JiangZongKang
@contact: top_jzk@163.com
@file: cnn_torch.py
@time: 2018/9/16 9:44

"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision

epoch = 4
batch_size = 50
lr = 0.001

# 获取训练集dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

# 打印MNIST数据集的训练集中特征值和目标值的尺寸
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# print(train_data.train_data[0])
# print(train_data.train_labels[0])

# 通过torchvision.datasets获取的dataset格式可直接可置于DataLoader
train_loader = data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)

# 获取测试集dataset
test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.ToTensor())

# 打印MNIST数据集的测试集中特征值和目标值的尺寸
# print(test_data.test_data.size())
# print(test_data.test_labels.size())

# 取前全部10000个测试集样本
