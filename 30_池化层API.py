"""
池化层（Pooling Layer）是卷积神经网络（CNN）中用于压缩特征图空间尺寸、保留重要特征的网络层。
池化层会将输入特征图划分为多个局部区域，再按照某种规则提取区域中的代表值。和卷积操作一样
常见的池化方式有两种：
    1、最大池化（Max Pooling）：取区域中的最大值。
    2、平均池化（Average Pooling）：取区域中的平均值。
池化之前同样可以进行填充Padding
注意：卷积会改变通道数，但是池化不会

案例：演示池化层api
"""

#导包
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#演示单通道池化
#创建单通道3*3图片张量
im1=torch.tensor([
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ]
])
print(im1,im1.shape)

#创建最大池化层
#参数1：池化核大小，参数2：步长，参数3：填充大小
pool1=torch.nn.MaxPool2d(2,1,0)
output1=pool1(im1)
print(output1,output1.shape)

#创建平均池化层
pool2=torch.nn.AvgPool2d(2,1,0)
output2=pool2(im1)
print(output2,output2.shape)

#演示多通道池化
im2=torch.randint(1,10,size=(3,4,4))
print(im2,im2.shape)

pool3=torch.nn.MaxPool2d(2,1,0)
output3=pool3(im2)
print(output3,output3.shape)