"""
二分类任务类似多分类，只是激活函数采用sigmoid
损失计算：
        L=-ylog(yi)-(1-y)log(1-yi)
其中：
    L：损失
    y：真实样本
    yi：预测概率
主要：因为公式中没有包含sigmoid函数，使用BCELoss时需要手动指定sigmoid

案例：演示二分类损失
"""
import torch
import torch.nn as nn

y_true=torch.tensor([1.,0,1,1],dtype=torch.float)
y_pre=torch.tensor([0.2,0.1,0.85,0.9],dtype=torch.float,requires_grad=True)

criterion_mean=nn.BCELoss()      #默认平均损失
loss_mean=criterion_mean(y_pre,y_true)
print(f'平均损失：{loss_mean}')

criterion=nn.BCELoss(reduction='none')  #各自的损失
loss=criterion(y_pre,y_true)
print(f'各自的损失：{loss}')
