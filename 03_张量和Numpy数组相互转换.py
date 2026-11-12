"""
一、将张量对象转换为numpy数组
    1、张量对象.numpy()，相当于赋值，共享内存，不创建对象，只增加一个引用
    2、张量对象.numpy().copy(),浅拷贝，只有一层时不共享内存、
二、numpy数组转换为张量对象
    1、from_numpy()，共享内存
    2、torch.tensor(np数组)，不共享内存
三、从张量对象提取内容
    标两张量.item()
"""
#导包
import torch
import numpy as np

#定义函数，演示张量转numpy数组
def dm01():
    n1=torch.tensor([1,2,3,4,5,6],dtype=torch.int32)  #小写只能根据指定数据创建张量
    print(n1)
    n2=n1.numpy()   #共享内存
    n3=n1.numpy().copy()    #不共享内存
    n1[0]=10
    print(n2)
    print(n3)

#定义函数，演示numpy数组转张量
def dm02():
    n1=np.arange(0,10,1)
    print(n1)
    n2=torch.from_numpy(n1)     #共享内存
    n3=torch.tensor(n1)         #不共享内存
    n1[0]=10
    print(n2)
    print(n3)

#定义函数，演示item()，
def dm03():
    n1=torch.tensor(1)
    n2=torch.tensor([1,2])
    a=n1.item()
    # b=n2.item()       #不是标量张量
    print(a)
    # print(b)

#测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()