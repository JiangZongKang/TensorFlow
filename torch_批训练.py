#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author: JiangZongKang
@contact: top_jzk@163.com
@file: torch_批训练.py
@time: 2018/9/20 16:49

"""
import torch
import torch.functional as f
import torch.utils.data as data

batch_size = 5
x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_dataset = data.TensorDataset(x,y)
loader = data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        print('Epoch:',epoch,'Step:',step,'batch_x:',batch_x.numpy(),'batch_y:',batch_y.numpy())
