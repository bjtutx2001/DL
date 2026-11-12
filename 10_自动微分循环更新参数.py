"""
案例：演示自动微分循环更新参数

需求：
    求y=2*x**2的极小值点，并打印y是极小值时的梯度值

步骤：
    1、初始点x=10
    2、定义函数y=2*x**2
    3、利用梯度下降，循环迭代100次
    4、正向计算（前向传播）
    5、梯度清零
    6、反向传播
    7、梯度更新
总结：在 PyTorch 中，对张量设置 requires_grad=True，本质上是告诉 PyTorch：
    后续如果使用这个张量参与可微分运算，需要追踪相关计算过程，以便通过反向传播计算梯度。
    设置了自动微分的张量为什么不能直接转换成 NumPy 数组，主要是因为：NumPy 不支持 PyTorch 的自动微分机制，直接转换会让张量脱离 PyTorch 的计算图。
    如果想转换，需要借助detach(),detach()会与原张量共享内存
"""
import torch
import copy

# 1、初始点x=10
x=torch.tensor([10],requires_grad=True,dtype=torch.float)
# 2、定义函数y = 2 * x ** 2
y=2 * x ** 2
lr=0.01
# 3、利用梯度下降，循环迭代100次
for i in range(1,101):
    # 4. 梯度清零
    if x.grad is not None:
        x.grad.zero_()
    # 5、正向计算（前向传播）
    y = 2 * x ** 2
    # 6、反向传播
    y.backward()
    # 7、梯度计算
    x_grad=copy.deepcopy(x.grad)
    # 8、更新参数
    with torch.no_grad():       #更新 x 时，PyTorch 不记录参数更新过程，避免把优化步骤加入计算图。
        # x=x-lr*x_grad   创建新的张量，并让原来的张量指向他，自动微分特性会变
        x -= lr * x.grad    #对原来的张量进行原地修改
print(x,x_grad)

#演示detach()
b1=torch.tensor([10],requires_grad=True,dtype=torch.float)
# b2=b1.numpy()     #报错，设置了自动微分转不了
b2=b1.detach().numpy()
b1.data[0]=20
print(b1,b2)    #一样，共享内存


