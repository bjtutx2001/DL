"""
张量：
    介绍：存储统一类型元素的容器，元素必须是数值才可以，在Pytorch框架中，张量是处理和存储数据的基本方式
    创建方式：
        1、torch.tensor  根据指定数据创建张量，可以指定元素类型
        2、torch.Tensor  根据指定形状、数据创建张量,不支持dtype方式指定数据类型，需要用to（torch.int64）
        3、torch.IntTensor、torch.FloatTensor、torch.DoubleTensor  创建指定形式的张量
"""
#导包
import torch
import numpy as np

#定义函数：演示张量创建
def dm01():
    #创建标量张量，默认元素类型为float32
    t1=torch.tensor(10)
    print(f't1:{t1}，type{type(t1)}')
    #列表转成张量
    data=[[1,2,3],[4,5,6],[7,8,9]]
    t2=torch.tensor(data,dtype=torch.float)
    print(f't2:{t2}，type{type(t2)}')
    #numpy数组转成张量
    t3=np.random.randint(0,10,size=[5,5])
    print(t3)
    t4=torch.tensor(t3)
    print(f't4:{t4}，type{type(t4)}')
    #创建指定维度的张量，例如两行三列，报错
    # t5=torch.tensor(2,3)
    # print(t5)
    #创建指定维度张量，要用Tensor
    t5=torch.Tensor(2,3)
    print(f't5:{t5}')

    t6=torch.IntTensor([1,2,3,4,5,6])
    print(f't6:{t6}')

    #创建指定值张量
    #全1
    t7=torch.ones(5,5)  #5*5全1张量
    print(f't7:{t7}')
    t8=torch.ones_like(t4)  #创建格式和t4一样的全1张量
    print(f't8:{t8}')
    #全0
    t9=torch.zeros(5,5)
    print(f't9:{t9}')
    t10=torch.zeros_like(t9)
    print(f't10:{t10}')
    #指定值
    t11=torch.full((2,3),255)
    print(f't11:{t11}')
    t12=torch.full_like(t11,254)
    print(f't12:{t12}')
    #空
    t13=torch.empty(3,3)
    print(f't13:{t13}')

#测试
if __name__ == '__main__':
    dm01()