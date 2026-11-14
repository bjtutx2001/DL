"""
损失函数是衡量模型参数的质量的函数，通常与预测值与真实值有关，也叫成本函数、代价函数、误差函数
分类：
    1、分类问题：
        多分类：多分类交叉熵损失
        二分类：二元交叉熵孙淑
    2、回归问题：
        MAE
        MSE
        RMSE
        Smooth L1：综合MAE和MSE

多分类交叉熵损失：用于衡量模型预测的类别概率分布与真实类别之间的差距。模型给真实类别分配的概率越高，损失越小。模型给真实类别分配的概率越低，损失越大。
计算公式：
             C
        L = -Σyi*ln(Pi)
            i=1
其中：
    C：类别总数
    yi：真实标签的One-Hot编码，真实类别为1，其余为0
    pi：模型预测属于第i类的概率，由SoftMax求出
    L：单个样本的交叉熵损失
为什么需要 Softmax？
    神经网络最后一个全连接层通常输出的是原始分数（Logits），这些分数不一定在0到1之间，也不一定相加等于1，通过SoftMax可以把原始分数转换成类别概率，在计算交叉熵
    PyTorch 的 nn.CrossEntropyLoss() 已经在内部结合了 LogSoftmax 和 NLLLoss，所以训练时不需要手动添加 Softmax。

案例：演示多分类损失
"""
import torch
import torch.nn as nn

#演示计算交叉熵损失
#手动创建真实值和预测值
y_true=torch.tensor([[1,0,0],[0,0,1]],dtype=torch.float)  #真实值
y_pred=torch.tensor([[0.1,0.1,0.8],[0.1,0.2,0.7]],requires_grad=True,dtype=torch.float)  #预测值

#创建多元交叉熵损失函数
criterion = nn.CrossEntropyLoss()   #默认采用平均损失
loss = criterion(y_pred,y_true)
print(f'损失：{loss}')