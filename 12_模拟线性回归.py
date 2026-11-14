"""
案例：pytorch模拟线性回归

TensorDatase：数据集对象，将特征张量和标签张量按照样本索引一一对应地封装成数据集，方便后续模型训练。
DataLoader：数据加载器，可以按照指定批次加载数据（一次取几个），具有打乱、多进程加载功能
"""

#导包
import torch
from sklearn.datasets import make_regression        #线性回归数据集
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch import nn,optim

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


#定义函数，创建线性回归数据样本
def data_creater():
    x,y,coef=make_regression(
        n_samples=100,      #样本数
        n_features=1,       #特征数
        noise=10,           #噪声大小
        coef=True,          #是否返回系数，默认False
        random_state=37     #随机种子
    )
    # print(x,y)
    x=torch.tensor(x,dtype=torch.float)
    y=torch.tensor(y,dtype=torch.float)
    return x,y,coef

#定义函数，表示训练模型
def train(x,y,coef):
    #创建数据集对象，将特征与标签一一对应，封装到一起
    dataset=TensorDataset(x,y)
    # print(dataset[0])

    #创建数据加载器对象，制定每次取得数据个数
    #参数1：数据源，参数2：每轮取得个数，参数3：每轮是否打乱数据
    dataloader=DataLoader(dataset,batch_size=16,shuffle=True)

    #创建初始线性回归模型
    #位置参数：样本特征维度，样本标签维度，是否包含可训练偏置，
    #注意：创建模型时会自动初始化权重和偏置，因此后续能直接进行预测
    estimator = nn.Linear(1,1,True)     #nn.Linear会把权重和偏置注册成模型参数，方便统一管理，且默认已经为参数设置了 requires_grad=True。

    #创建损失函数对象
    criterion = nn.MSELoss()

    #创建优化器对象
    optimizer = optim.SGD(estimator.parameters(),lr=0.01)       #SGD表示随机梯度下降优化器，estimator.parameters()表示所有可训练参数

    #定义变量，分别表示：训练轮数，每轮的平均损失，训练总损失，样本批次
    epochs,loss_list,total_loss,total_sample = 100,[],0.0,0
    for epoch in range(epochs):
        #分批次训练
        for x_train,y_train in dataloader:  #每批次16个
            y_pre=estimator(x_train)        #预测
            loss=criterion(y_pre,y_train)   #计算损失
            total_loss += loss.item()       #.item()能获取张量的原生类型的数据，只能用于标量（一个元素）
            total_sample += 1
            #梯度清零+反向传播+梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




#测试
if __name__=='__main__':
    x,y,coef=data_creater()
    train(x,y,coef)