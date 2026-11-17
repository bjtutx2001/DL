"""
神经网络的正则化（Regularization）与传统机器学习中的正则化，本质上是相同的，都是为了防止模型过拟合，提高模型的泛化能力。
统机器学习通常采用 L1、L2 正则化，而神经网络除了 L1、L2 正则化，还经常采用 Dropout(随机失活)、Early Stopping（早停法）、数据增强等方法。
1、Dropout正则化：
    描述：在训练过程中，按照一定概率随机将部分神经元的输出置为0，使这些神经元暂时不参与当前这次前向传播及对应的反向传播，训练过程可以视为
         对完整神经网络的一些子网络进行训练
    目的：减少神经元之间的过度依赖，提高泛化能力。
    Dropout原则上只在训练工程中生效，测试阶段可通过 model.eval() 关闭 Dropout 的随机失活行为。
    PyTorch 默认采用的是 Inverted Dropout（反向缩放 Dropout），训练时计算公式为：
            a’=r*a/(1-p)
            a:神经元原来的输出
            p：失活概率，一般不超过0.5
            r：随机掩码，取值为0,1，概率分别为p和1-p
            a'：Dropout处理后的输出
            除以1-p能够保证每个神经元输出的数学期望保持不变。

"""
import torch
import torch.nn as nn

#演示随机失活
torch.manual_seed(37)
#定义输入
x = torch.randn(2,3)*2+5    #均值为5，标准差为2的3*3随机输入
print(f'输入{x}')
#定义隐藏层1，线性回归
linner=nn.Linear(3,3,bias=False)    #权重使用 Kaiming Uniform 初始化,偏置使用均匀分布初始化,这里没有启用偏置
#加权求和
l1=linner(x)
print(f'加权结果：{l1}')
#激活函数
output=torch.relu(l1)   #Relu负数变0，正数不变
print(f'激活结果：{output}')
#对激活结果进行随机失活处理
dropout=nn.Dropout(0.5)     #失活概率为0.5
d1=dropout(output)
print(f'失活结果：{d1}')
