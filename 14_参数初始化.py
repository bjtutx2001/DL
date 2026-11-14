"""
在深度学习中，参数初始化（Parameter Initialization） 是指在神经网络正式训练之前，
为模型中的可训练参数（主要是权重 weight 和偏置 bias）设置初始值。
初始化的主要目标有三个：
    1、打破对称性：避免同一层的多个神经元学习出完全相同的特征。
    2、控制信号大小：避免前向传播过程中激活值越来越大或越来越小。
    3、改善梯度传播：降低反向传播过程中梯度爆炸或梯度消失的风险。
常见的初始化方式包括：
    1、零初始化（Zero Initialization）：偏置常用全0
    2、全1初始化
    3、随机正态初始化
    4、随机均匀初始化：默认区间为（0,1）,通常采用（-1/sqrt(d),1/sqrt(d),d为输入神经元个数）
    5、Xavier 初始化（Glorot Initialization）：泽维尔，激活函数非ReLu，优先选这个
            正态(服从N(0,std^2),std=gain*sqrt(2/(fan_in输入神经元个数+fan_out输出神经元个数)))，均匀（[-limit,limit],limit=gain*sqrt(6/(fan_in+fan_out))）
    6、Kaiming 初始化（He Initialization）：由何恺明发明，优先选择ReLu激活函数
            又可以分为正态（范围[0,std],std=sqrt(2/输入神经元数fan_in)）和均匀(范围[-limit,limit]，limit=sqrt(6/输入神经元数))
    7、正交初始化（Orthogonal Initialization）：权重矩阵满足正交特性
    8、预训练权重初始化（Pretrained Initialization）
无法打破对称性：固定值初始化，浅层网络优先选固定值初始化

演示参数初始化
"""
import torch.nn as nn

#均匀分布随机初始化
linear1=nn.Linear(5,4)  #输入维度5，输出维度4
nn.init.uniform_(linear1.weight,a=1,b=2)    #对权重weight进行初始化，均匀分布区间为[1,2)
nn.init.uniform_(linear1.bias)              #对偏置bias进行初始化，默认区间[0,1)
print(linear1.weight)
print(linear1.bias.data)

#固定值初始化
linear2=nn.Linear(5,4)
nn.init.constant_(linear2.weight,1)     #权重全1
nn.init.constant_(linear2.bias,0)       #偏置全0
print(linear2.weight)
print(linear2.bias.data)

# 全1初始化
linear2=nn.Linear(5,4)
nn.init.ones_(linear2.weight)       #全1
nn.init.zeros_(linear2.bias)        #全0
print(linear2.weight)
print(linear2.bias.data)

#正态初始化
linear2=nn.Linear(5,4)
nn.init.normal_(linear2.weight,1,0)     #均值1，标准差0，正态
nn.init.normal_(linear2.bias,0,2)       #均值0，标准差2，正态
print(linear2.weight)
print(linear2.bias.data)

#kaiming初始化，指定的张量维度必须是二维，因为要有输入和输出
linear2=nn.Linear(5,4)
nn.init.kaiming_normal_(linear2.weight,mode="fan_out")       #kaiming正态，服从N(0,σ^2),方差计算选输出神经元，σ=2/fan_out
# nn.init.kaiming_uniform_(linear2.weight,mode="fan_in")       #kaiming均匀，边界计算选输入神经元，sqrt（6/fan_in）
nn.init.zeros_(linear2.bias)                                #0初始化偏置
print(linear2.weight)
print(linear2.bias.data)

#xavier初始化，指定的张量维度必须是二维，因为要有输入和输出
linear2=nn.Linear(5,4)
nn.init.xavier_normal_(linear2.weight,gain=1)       #xavier正态，服从N(0,σ^2),σ=gain*sqrt(2/(in+out))
# nn.init.xavier_uniform_(linear2.weight,gain=2)    #xavier标准，边界计算：gain*sqrt（6/(in+out)）
nn.init.ones_(linear2.bias)                         #全1初始化偏置
print(linear2.weight)
print(linear2.bias.data)