"""
张量基本运算API：
    add()   加法
    sub()   减法
    mul()   乘法
    div()   除法
    neg()   取反
也可以使用+、-、*、/

张量的点乘：相同形状的张量进行对应元素相乘
    t1*t2
    t1.mul(t2)

张量的矩阵乘法：满足基本格式
    t1@t2
    t1.matmul(t2)
    t1.dot(t2)      #只针对一维，对维有特殊运算规则
"""

import torch

t1=torch.tensor([1,2,3])
t2=t1+10
t3=t1.add(5)
print(t2)
print(t3)
t1.add_(1)      #add_()会修改源数据
print(t1)
t1+=10
print(t1)
#其他运算效果同上
