"""
输出特征图尺寸计算公式：
out=（in+2P-f）/s,整体加1，取整数

案例；演示卷积层api
"""
#导包
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#读取真彩图片
img1=plt.imread('./data/狼主.png')
#打印图片信息
# plt.imshow(img1)
# plt.show()
print(img1.shape)   #结果为HWC，通道在最后一个维度

#把图片从HWC变成CHW，这样才能进行处理
#封装成张量
im2=torch.tensor(img1,dtype=torch.float32)
#变成CHW
# HWC 是把每个像素的RGB值放在一起，而 CHW 是把所有像素的R值放在一起、G值放在一起、B值放在一起,
# 使用permute并不会改变像素点在内存中存储的顺序，只是读取的索引变了。
im2=im2.permute(2,0,1)
print(im2.shape)

#增加维度，表示图片数量，才能处理
im2=im2.unsqueeze(0)    #在第0维增加一个维度
print(im2.shape)

#创建卷积层对象，提取特征图
#参数1：输入通道数，参数2：输出通道数，参数3：卷积核尺寸，参数4：步长，通道5：填充
con=nn.Conv2d(in_channels=4,out_channels=5,kernel_size=(3,3),stride=(1,1),padding=0)

#对原图进行卷积
im3=con(im2)
print(im3.shape)

#查看提取到的特征图
#把图片降维成CHW
im3=im3.squeeze(0)  #不写参数为删除大小为1的纬度，写参数为删除第0维
print(im3.shape)

#CHW变成HWC
im4=im3.permute(1,2,0)
print(im4.shape)

#可视化四个通道,要转成numpy数组
# PyTorch的Tensor 可以参与自动求导，而NumPy不支持PyTorch的自动求导机制。detach()的作用就是将张量从当前计算图中分离出来，使其不再参与梯度计算。
# detach()返回一个新的张量对象，但通常与原张量共享底层存储。
plt.imshow(im4[:,:,0].detach().numpy())
plt.show()
plt.imshow(im4[:,:,1].detach().numpy())
plt.show()
plt.imshow(im4[:,:,2].detach().numpy())
plt.show()
plt.imshow(im4[:,:,3].detach().numpy())
plt.show()