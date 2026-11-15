"""
Smooth L1 Loss（平滑 L1 损失） 结合了 L1 Loss（平均绝对误差） 和 L2 Loss（平方误差） 的特点。
定义：
    L(e)=e^2/2β     e<β
    L(e)=e-β/2      e>=β
其中，e为绝对误差，β是控制损失切换的参数，通常取1
API:SmoothL1Loss()


"""
import numpy as np
import torch
import torch.nn as nn

y_true=torch.tensor([0,1,2,3,4,5,6,7,8,9],dtype=torch.float)
y_pred=torch.tensor([0.3,1.4,2.4,3.4,4.2,5.3,6.6,7.8,8.3,9.6],dtype=torch.float,requires_grad=True)
criterion=nn.SmoothL1Loss()
loss_L1=criterion(y_pred,y_true)
print(loss_L1)