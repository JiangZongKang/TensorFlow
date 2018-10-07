#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author: JiangZongKang
@contact: top_jzk@163.com
@file: test_atom.py
@time: 2018/9/15 9:43

"""

'''
import tensorflow as tf
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as f
import torchvision
from torch.autograd import Variable
'''

'''
net = nn.Sequential(
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,2)
)

optimizer = torch.optim.SGD(params=net.parameters(),lr=0.5)
loss = nn.MSELoss()
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
print(type(x))
y = torch.linspace(10,1,10)
print(type(y))
'''

'''
x = tf.constant(-1.0)
y = tf.abs(x)
with tf.Session() as sess:
    sess.run(y)
    print(y)
'''

# for i in range(1,6):
#     print(i)


a = {'name':'laojiang','age':18}
print(a['name'])