"""
图像分类：
    1、二值图
    2、灰度图
    3、索引图：像素值是颜色的索引，有个索引表会将像素值与颜色一一对应
    4、RGB真彩图
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#绘制全黑、全白图
# shape：HWC 高度宽度通道
img1=np.zeros((200,200,3),dtype=np.uint8)
print(img1)
plt.imshow(img1)
# plt.show()
img2=torch.full((200,200,3),255,dtype=torch.uint8)
plt.imshow(img2)
plt.show()