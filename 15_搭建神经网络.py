"""
深度学习的四个步骤：
    1、准备数据
    2、搭建神经网络
    3、模型训练
    4、模型测试

神经网络搭建流程：
    1、定义类，继承自nn.Module
    2、在_init_方法中，搭建神经网络
    3、在forward()方法中，完成前向传播

补充：当一个对象被当作函数调用时，就会自动触发该对象所属类定义的__call__，简单理解就是只要向对象传递参数，就会触发
下文的前向传播forward正是定义在__call__方法内的一个类方法，传入参数——>触发__call__——>触发forward

案例：演示搭建神经网络
"""
import torch
import torch.nn as nn
from torchsummary import summary    #查看模型结构和参数信息的工具。

#搭建神经网络，继承自nn.Module
class Net(nn.Module):
    #初始化魔法方法
    def __init__(self):
        super(Net,self).__init__()  #调用父类的初始化方法

        #搭建神经网络，隐藏层+输出层
        #隐藏层1：输入特征3，输出特征3
        self.linear1 = nn.Linear(3,3)   #线性回归神经网络，3*4个参数
        #隐藏层2：输入特征3，输出特征2
        self.linear2 = nn.Linear(3,2)   #线性回归神经网络,2*4个参数
        #输出层：输入特征2，输出特征2
        self.output = nn.Linear(2,2)   #线性回归神经网络，2*3个参数

        #对隐藏层进行参数初始化，实际上在定义隐藏层的时候会自动实现，这里做演示
        nn.init.xavier_normal_(self.linear1.weight)     #隐藏层1权重做泽维尔标准化
        nn.init.zeros_(self.linear1.bias)               #隐藏层1偏置做全0初始化
        nn.init.kaiming_normal_(self.linear2.weight)    #隐藏层2权重做kaiming标准化
        nn.init.zeros_(self.linear2.bias)               #隐藏层1偏置做全0初始化

    #实现前向传播
    def forward(self,x):    #x表示输入特征
        #第一层隐藏层计算加权求和和激活函数
        # x=self.linear1(x)
        # x=torch.sigmoid(x)
        x=torch.sigmoid(self.linear1(x))    #合并写法，

        #第二层
        x = torch.relu(self.linear2(x))

        #输出层
        x=torch.softmax(self.output(x),dim=-1)  #dim=-1 表示沿着张量的最后一个维度(列)进行 Softmax 计算。这里可以理解为行代表样本，列代表特征，按列计算就是把样本的特征进行加权计算，使最终分类结果和为1

        #返回结果
        return x
#训练模型
def train():
    #创建模型对象
    my_model=Net()
    # print(f'my_model:{my_model}')

    #创建样本数据，随机生成
    data=torch.randn(size=(4,3))
    print(data.requires_grad)   #自建数据默认没有自动微分

    #调用神经网络，进行模型训练
    output=my_model(data)
    print(f'output:{output}')
    print(f'output.requires_grad:{output.requires_grad}')

    #计算和查看模型参数
    #参数1模型对象，参数2：输入数据维度，通常为单个样本的形状（特征数）,查看模型参数信息，不涉及计算
    #有一个torchinfo包，也是查看模型参数信息的，input_size输入要求为输入数据维度
    summary(my_model,input_size=(3,))       #会直接输出模型层数和参数个数等信息

    for name, param in my_model.named_parameters():
        print(f'name:{name}\n')
        print(f'param:{param}')


#测试
if __name__=='__main__':
    train()


