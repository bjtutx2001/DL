"""
平均绝对误差损失：L1Loss，L1曼哈顿距离
均方误差损失：L2Loss，L2欧氏距离
"""
import torch
import torch.nn as nn

y_true=torch.tensor([1,2,3,4,5],dtype=torch.float)
y_pre=torch.tensor([1,1.5,1.9,2.9,4.7],dtype=torch.float,requires_grad=True)
criterion1=nn.L1Loss(reduction='none')    #各自的
loss1=criterion1(y_pre,y_true)
criterion2=nn.L1Loss(reduction='sum')    #总的
loss2=criterion2(y_pre,y_true)
criterion3=nn.L1Loss(reduction='mean')    #平均的
loss3=criterion3(y_pre,y_true)
print(f'各自的:{loss1},\n平均的：{loss3},\n总的:{loss2}')

criterion4=nn.MSELoss()
loss4=criterion4(y_pre,y_true)
print(f'\n{loss4}')
