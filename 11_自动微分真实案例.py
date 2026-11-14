"""
案例：演示自动微分的真实应用场景

广播机制的判断规则:
    1. 从两个张量形状的最后一个维度开始，向前逐维比较。
    2. 对应维度相等，或者其中一个为 1，就可以兼容。
    3. 如果维度数量不同，较短的形状会在左侧补 1 后再比较。
    4. 如果某个对应维度既不相等，也没有一个为 1，则无法广播。
"""
import torch

#定义特征
x=torch.ones(2,5)

#定义标签
y=torch.zeros(2,4)

#根据公式y=wx+b，在进行矩阵乘法的时候，w维度应该是：行数为x的列数5，列数为y的列数4
w=torch.randn(5,4,requires_grad=True)   #初始化权重
print(f'{x},\n{y},\n{w}')

#初始化偏置
b=torch.randn(4,requires_grad=True)    #根据广播原则，会自动扩展为和y一样的维度（2,4）

#前向传播，计算预测值
y_pre=x @ w + b
print(y_pre)

#定义损失函数
criterion=torch.nn.MSELoss()        #nn是neural network神经网络，pytorch提供了很对的损失函数供选择
loss=criterion(y_pre,y)         #计算损失

#反向传播，进行自动微分，后续进行参数更新
loss.backward()           #多个变量，要先求和,这里由于指定为MSE损失函数，已经进行了求和,结果为标量张量

#查看w，b的梯度
print(w.grad)
print(b.grad)
