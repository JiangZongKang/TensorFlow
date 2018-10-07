#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@author: JiangZongKang
@contact: top_jzk@163.com
@file: test_list.py
@time: 2018/9/21 7:24

"""

a = [1,2,3,1,2]
addr_index = [x for x in range(len(a)) if a[x] == 1]
print(addr_index)
# print(len(a))

for w in enumerate(a):
    # print(w[1])
    if w[1] == 1:
        print(w[0])


