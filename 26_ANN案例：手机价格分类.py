"""
基于手机的20列特征，预测手机的价格区间，用深度学习完成，本质分类问题
实现步骤：
    1、构建数据集
    2、搭建神经网络
    3、模型训练
    4、模型测试

优化思路：
    1、SGD——>Adam
    2、调整学习率
    3、对数据进行标准化
    4、增加隐藏层数
    5、调整训练轮次
"""
#导包
import torch
from torch.utils.data import TensorDataset, dataloader  # 数据集对象 数据——>Tensor——>数据集——>数据加载器
from torch.utils.data import DataLoader     #数据加载器
import torch.nn as nn
import torch.optim as optim                 #优化器
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from torchsummary import summary

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#定义函数，构建数据集
def create_dataset():
    # 加载csv文数据集
    data=pd.read_csv('./data/train.csv')
    # print(data.head())
    # data.info()     #2000行，21列

    #获取特征和标签
    x,y=data.iloc[:,:-1],data.iloc[:,-1]
    #把特征转成浮点型
    x=x.astype('float32')
    y=y.astype(np.float32)
    # print(x.dtypes)

    #划分数据集
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=37,stratify=y)

    #把数据集转换成张量数据集，思路：数据——>张量数据——>数据集TensorDataset——>数据加载器DataLoader
    #源数据是DF对象，需要先转成numpy数组或获取值对象，并且进行多分类时推荐采用long数据类型

    # x_train=torch.tensor(x_train.to_numpy())
    # x_test=torch.tensor(x_test.values)
    # y_train=torch.tensor(y_train.values)
    # y_test=torch.tensor(y_test.values)     #转张量
    # # print(x_train.shape[1])
    # # print(y_train.unique())
    #
    # dataset_train=TensorDataset(x_train,y_train)
    # dataset_test=TensorDataset(x_test,y_test)   #转数据集

    dataset_train=TensorDataset(torch.tensor(x_train.to_numpy()), torch.tensor(y_train.to_numpy()))
    dataset_test=TensorDataset(torch.tensor(x_test.values),torch.tensor(y_test.values))   #转数据集


    dataloader_train=DataLoader(dataset_train,batch_size=100,shuffle=True)    #shuffle是否打乱
    dataloader_test=DataLoader(dataset_test,batch_size=100,shuffle=True)

    # print(len(y.unique()))

    return dataloader_train,dataloader_test,dataset_train,dataset_test,x_train.shape[1],len(y.unique())
                                                                                #特征数            #标签数、

#创建神经网络，三层
class phone_price(nn.Module):
    def __init__(self,x_dim,y_dim):             #输入维度和输出维度
        super(phone_price,self).__init__()      #初始化父类属性，可以省略参数
        self.x_dim=x_dim
        self.y_dim=y_dim
        #搭建神经网络
        self.linear1=nn.Linear(self.x_dim,128)         #隐藏层1
        self.linear2=nn.Linear(128,256)         #隐藏层2
        self.output=nn.Linear(256,self.y_dim)          #输出层

    #定义前向传播
    def forward(self,x):    #x是特征数据
        #隐藏层1：加权求和+激活函数(relu)
        x=torch.relu(self.linear1(x))
        #隐藏层2：同上
        x=torch.relu(self.linear2(x))
        #输出层，激活函数采用SoftMax
        # output=torch.softmax(self.output(x),dim=-1)     #这是正常写法，但是不需要，因为多分类交叉熵损失函数包括了Softmax计算
        output = self.output(x)
        return output

#定义函数，训练模型
def train_model(dataloader_train,x_dim,y_dim):
    #创建神经网络模型
    model=phone_price(x_dim,y_dim)
    #定义损失函数，多分类选择多元交叉损失熵CrossEntropyLoss，要求预测结果为浮点型，真实数据为long
    criterion = nn.CrossEntropyLoss()   #默认计算所有样本的平均损失
    #创建优化器对象，采用随机梯度下降
    # print(model.parameters())
    optimizer=optim.SGD(model.parameters(),lr=0.001) #model.parameters()为权重和偏置参数

    #训练模型，50轮，每轮1600个数据，每批100条
    epochs=50   #轮数
    for epoch in range(epochs):
        #定义变量，记录每次的训练损失和训练批次
        totle_loss,batch_num=0.0,0
        start=time.time()   #标记训练开始时间
        #开始本轮各个批次的训练
        for x,y in dataloader_train:
            model.train()   #训练模式，model。eval测试模式
            y_pred=model(x)     #预测
            loss=criterion(y_pred,y.long())    #计算损失
            optimizer.zero_grad()       #梯度清零
            loss.backward()             #反向传播
            optimizer.step()            #更新参数
            totle_loss += loss.item()   #累加损失
            batch_num += 1              #累计批次

        #打印训练信息                                                     #：。4f 保留四位有效数字
        print(f'轮数：{epoch+1},总损失：{totle_loss}，平均批次损失：{totle_loss/batch_num:.4f}，训练时间：{time.time()-start:.2f}s')

    #所有轮次结束，保存模型参数
    #参数1：模型参数词典，参数2：保存路径
    torch.save(model.state_dict(),'./model/phone_price.pth')    #后缀名用pth、pkl、pickle均可
    print(f'保存成功！')

#定义函数，训练模型
def test_model(dataloader_test,x_dim,y_dim):
    # 创建神经网络分类对象
    model=phone_price(x_dim,y_dim)
    #加载模型参数
    model.load_state_dict(torch.load('./model/phone_price.pth'))
    #定义变量，记录预测正确的样本个数
    correct=0
    #从数据加载器获取数据
    for x,y in dataloader_test:
        model.eval()
        y_pred=model(x) #由于定义神经网络的时候没有对输出使用softmax，得到的预测结果只是预测分数，并不是预测概率，用argmax()获取分数最大值对应的下标，就是类别
        # print(y_pred)
        #用argmax()查找最大值
        y_pred=torch.argmax(y_pred,dim=1)   #dim=1表示列，即要进行归约（消除）的是列维度。也就是按行进行查找，每行的最大值
        #打印每个样本的预测值,学习率过小可能会全部预测成一类值，需要调小学习率
        print(y_pred)
        print(y)
        #统计正确样本
        print((y_pred==y).sum())    #y_pre==y输出bool值，sum()统计True的个数
        correct += (y_pred==y).sum()
    #打印准确率
    print(f'正确率为：{correct/len(dataloader_test.dataset)}')

#测试
if __name__ == '__main__':
    #读取数据
    dataloader_train,dataloader_test,dataset_train,dataset_test,x_dim,y_dim=create_dataset()
    #构建神经网络
    model = phone_price(x_dim,y_dim)
    #计算模型参数
    summary(model,input_size=(x_dim,))
    #训练模型
    train_model(dataloader_train,x_dim,y_dim)
    #测试网络
    test_model(dataloader_test,x_dim,y_dim)


