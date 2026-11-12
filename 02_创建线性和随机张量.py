"""
案例：演示pytorch中创建线性和随机张量
涉及函数：
    torch.arange()
    torch.linspace()
    torch.random.initial_seed()
    torch.random.manual_seed()
    torch.rand/randn()  均匀/正太
    torch.randint(low,high,size=())
"""
import torch

#创建指定范围线性张量
t1=torch.arange(0,10,2) #依旧左闭右开
print(t1)

#指定线性张量——>等差数列
#参数1：起始，参数2：结束，参数3：元素个数，
t2=torch.linspace(0,10,2)
print(t2)

#设置随机种子
torch.random.initial_seed()         #默认采用当前时间戳作为随机种子，常用于查看当前随机数生成器的随机种子状态
torch.random.manual_seed(37)       #设置随机种子的作用：指定随机数生成的开头

#创建随机张量
#均匀分布
t3=torch.rand(3,3) #均匀分布（0,1）
print(t3)
t4=torch.randn(3,3)    #正态分布
print(t4)
t5=torch.randint(0,10,(3,3))    #0到10之间整数
print(t5)

#创建好之后，进行类型转换
#用type
print(f't5:{t5},元素类型：{t5.dtype}')
t6=t5.type(torch.int32) #长整
print(f't6:{t6},元素类型：{t6.dtype}')
#用half(),float(),double(),short(),int(),long()
print(t6.half())
print(t6.float())       #float默认32
print(t6.double())
print(t6.short())
print(t6.int())
print(t6.long())        #int默认64

