"""
 在反向传播更新参数的时候，学习率的选择很重要：
    学习率较小时，会导致参数的更新很慢，梯度下降缓慢
    学习率较大时，会导致梯度震荡（梯度在正负之间来回变化）和梯度爆炸
之前的AdaGrad、RMSprop和Adam学习率采用自动更新的方式，除此以外，我们可以采用等间隔、指定间隔、指数等方式，来手动控制学习率
分类：
    1、等间隔学习率:适用于大数据集。较为简单的任务
    2、指定间隔学习率：对训练平稳性要求较高的任务
    3、指数学习率：高精度训练，避免过快收敛


"""
from sched import scheduler

#导包
import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#演示等间隔学习率
def dm01():
    #定义变量，记录初始的学习率，训练轮数，每轮训练的批次数
    lr,epochs,iteration=0.1,200,100

    #创建数据集y_true,x,w
    y_true=torch.tensor([0],dtype=torch.float)
    x=torch.tensor([1.0],dtype=torch.float)
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float)

    #创建优化器：动量法
    optimizer=torch.optim.SGD([w],lr=lr,momentum=0.9)
    #创建学习率调度对象
    #参数1：优化器对象，参数2：间隔轮数，参数3：衰减系数，每次调整时，学习率乘多少
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.50)

    #创建列表，分别保存训练轮数，本轮学习率
    lr_list,epoch_list,iter_list=[],[],[]

    #循环学习
    for i in range(epochs):
        epoch_list.append(i+1)
        lr_list.append(scheduler.get_last_lr()[0])   #获取最新一次的学习率

        #按批次循环学习
        for _ in range(iteration):
            y_pred=w*x
            loss=(y_pred-y_true)**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #每轮训练结束，更新学习率
        scheduler.step()

    # 打印结果
    print(f'轮数：{epoch_list}\n学习率：{lr_list}')
    plt.plot(epoch_list,lr_list,color='blue',label='等间隔变化')
    plt.xlabel('训练轮数')
    plt.ylabel('学习率')
    plt.title('学习率随轮数变化关系')
    plt.legend()
    plt.show()

#演示指定间隔学习率衰减
def dm02():
    lr,epochs,iteration=0.1,200,100
    #定义预测值，特征，权重
    y_true=torch.tensor([0],dtype=torch.float)
    x=torch.tensor([1.0],dtype=torch.float)
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float)

    #创建优化器：动量法
    optimizer=torch.optim.SGD([w],lr=lr,momentum=0.9)

    #创建学习率调度对象
    milestons=[60,120,180]  #定义变量，记录学习率变化的轮数
    schedular=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestons,gamma=0.5)

    #创建列表，记录学习率和轮次
    lr_list,epoch_list,iter_list=[],[],[]
    #循环训练，轮次
    for epoch in range(epochs):
        epoch_list.append(epoch+1)
        lr_list.append(schedular.get_last_lr()[0])
        # 循环训练，批次
        for _ in range(iteration):
            y_pred=w*x
            loss=(y_pred-y_true)**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #更新学习率
        schedular.step()

    #打印
    print(f'学习率：{lr_list}\n轮次{epoch_list}')
    plt.plot(epoch_list,lr_list,color='g',label='指定间隔变化')
    plt.legend()
    plt.show()

#演示指数学习率衰减
def dm03():
    lr,epochs,iteration=0.1,200,100
    y_true=torch.tensor([0],dtype=torch.float)
    x=torch.tensor([1.0],dtype=torch.float)
    w=torch.tensor([1.0],requires_grad=True,dtype=torch.float)
    optimizer=torch.optim.SGD([w],lr=lr,momentum=0.9)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)
    lr_list,epoch_list,iter_list=[],[],[]
    for epoch in range(epochs):
        epoch_list.append(epoch+1)
        lr_list.append(scheduler.get_last_lr()[0])
        for _ in range(iteration):
            y_pred=w*x
            loss=(y_pred-y_true)**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    print(f'学习率：{lr_list}\n轮数{epoch_list}')
    plt.plot(epoch_list,lr_list,color='r',label='指数学习率衰减')
    plt.legend()
    plt.show()

#测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()