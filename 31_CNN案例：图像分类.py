"""
案例：演示CNN图片分类
数据集：计算机视觉模块自带的CIFAR10数据集，包括6w张（32,32,3）的图片，5W张训练集，1W张测试集，10个分类，每个分类6K张
步骤：
    1、准备数据
    2、搭建卷积神经网络
    3、模型训练
    4、模型测试

卷积层参数个数计算公式：输入通道数*卷积核尺寸*卷积核个数+卷积核个数（偏置）
池化层通常没有参数

优化方案：
    1、增加卷积层输出通道
    2、增加全连接层参数数
    3、调整学习率
    4、增加训练轮数
    5、修改激活函数
    6、增加网络层数
"""
#导包
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
# from torchsummary import summary      #调用时，input_size输入单个样本维度
from torchinfo import summary           #调用时，input_size输入需要包括batch_size

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#每批次样本数
batch_size = 8
#准备数据集
def create_data():
    # CIFAR - 10数据集的存储方式与普通PNG、JPG图片不同，因此不能直接使用imread()读取整个CIFAR - 10数据集。
    # CIFAR10()返回的不是一张图片，也不是一个包含所有图片的Tensor，而是一个Dataset数据集对象。专门用于加载cifar-10数据的接口
    #参数1：路径，参数2：是否使用训练集，参数3：读取样本时，将图片转化为tensor，参数4：如果路径没数据，是否尝试下载
    train_data=CIFAR10(root='./data',train=True,transform=ToTensor())     #训练集
    test_data=CIFAR10(root='./data',train=False,transform=ToTensor())

    return train_data,test_data

#搭建卷积神经网络
class ImageNet(nn.Module):
    def __init__(self):
        super().__init__()      #初始化父类成员
        #搭建神经网络
        #第一层卷积层：输入3通道，输出6通道，卷积核大小：3*3，步长：1，填充：0
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1), padding=0)
        #第一层池化层，池化窗口：2*2，步长：2，填充：0
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),padding=0)
        #第二层卷积层：输入6 通道，输出16 通道，卷积核大小：3*3，步长：1，填充：0
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=0)
        # 第二层池化层，池化窗口：2*2，步长：2，填充：0
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),padding=0)
        #第一个全连接层，采用线性回归,卷积完之后输出的单通道尺寸为：6*6
        self.linear1 = nn.Linear(16*6*6, 120)
        # 第二个全连接层，输入120，输出84
        self.linear2 = nn.Linear(120, 84)
        #输出层：输入84，输出10（类别个数）
        self.output = nn.Linear(84, 10)

    #定义前向传播
    def forward(self, x):   #x为传入的数据
        # x = self.conv1(x)       #卷积
        # x = torch.relu(x)       #激活
        # x = self.pool1(x)       #池化

        #合并写法
        #第一层卷积
        x=self.pool1(torch.relu(self.conv1(x)))
        #第二层卷积
        x=self.pool2(torch.relu(self.conv2(x)))

        #卷积完之后的尺寸为：(batch_size,通道数，宽，高)，也就是(8,16,6,6)，在进行线性回归之前，要将数据展平成（batch_size,特征数）
        x=x.reshape(x.shape[0], -1)     #8行576列
        #第一层线性回归
        x=torch.relu(self.linear1(x))
        #第二层线性回归
        x=torch.relu(self.linear2(x))
        #输出层
        out=self.output(x)  #正常来书还需要进行激活，但是多分类的多元交差损失熵函数CrossEntropLoss()包含了softmax操作

        return out

#定义函数，实现模型训练
def train_model(train_data):
    # 指定第一张 GPU
    device = torch.device('cuda:0')
    # 创建模型并移动到 GPU
    #模型定义在训练循环外，每轮训练完成不会重置参数，因此，训练轮数增大效果会更好（也不能一直大，会过拟合）
    model = ImageNet().to(device)

    #创建数据加载器
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2,pin_memory=True,persistent_workers=True)
    #损失函数，多元交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 创建优化器对象
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #循环训练
    epochs=30
    for epoch in range(epochs):
        #定义变量，记录总损失，总的样本数据，预测正确样本数，训练开始时间
        total_loss,total_samples,total_correct,train_start_time = 0,0,0,time.time()
        #切换为训练模式
        model.train()
        #遍历加载数据
        for x,y in data_loader:
            #将数据移动带GPU设备，non_blocking=True，允许 CPU 在数据传输尚未完成时继续向下执行程序。通常配合 pin_memory=True一起使用
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            #预测
            y_pred = model(x)
            #计算损失
            loss = criterion(y_pred, y)
            #梯度清零+反向传播+各项参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #统计预测正确的样本个数，每轮清空
            #查看样本预测结果
            # print(y_pred)
            # print(torch.argmax(y_pred,dim=-1))      #dim=-1,沿着列计算，就是计算每行
            #两个torch张量比较，会得到torch.bool结果，还是张量
            # GPU张量和CPU张量最大的区别是数据存储的位置不同。将GPU张量转换为NumPy数组时，必须先将数据从GPU显存移动到CPU内存，
            # 因为普通NumPy数组不能直接存储在GPU显存中。
            #.item()会直接把GPU张量传回CPU，并变成python标量
            #如果对GPU张量直接使用.numpy，会报错，必须先.cpu()将数据传回cpu
            total_correct += (torch.argmax(y_pred,dim=-1) == y).sum().item()      #每批8个，预测对的可能只有5,6个
            total_samples += len(y)     #每批8个
            total_loss += loss.item()*len(y)    #定义损失函数的时候使用的是平均损失，乘以每轮个数就是每轮总损失

        #一轮训练完成，打印每轮信息
        print(f'轮次:{epoch+1},平均损失:{total_loss/total_samples:.5f},准确率:{total_correct/total_samples:.5f},本轮数训练时间:{time.time()-train_start_time:.2f}s')

    #保存模型信息
    torch.save(model.state_dict(),'./model/cifar_model.pth')
    print(f'保存成功')

#定义函数，实现模型测试
def evaluate_model(test_data):
    #创建模型，并移动到GPU
    device = torch.device('cuda:0')
    model = ImageNet().to(device)
    #加载模型参数，并切换到测试模式，模型创建好默认是训练模式
    model.load_state_dict(torch.load('./model/cifar_model.pth'))
    model.eval()
    #创建数据加载器对象
    data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2,pin_memory=True,persistent_workers=True)
    #定义变量，统计预测正确的样本个数，总样本数
    total_correct, total_samples = 0,0
    for x,y in data_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_pred = model(x)
        total_correct += (torch.argmax(y_pred,dim=-1) == y).sum().item()
        total_samples += len(y)
    print(f'正确率：{total_correct/total_samples:.5f}')

#测试
if __name__=='__main__':
    train_data,test_data=create_data()
    #不能直接打印，因为是数据集对象
    # print(train_data,test_data)
    print(f'训练集形状：{train_data.data.shape}')
    print(f'训练集类别：{train_data.class_to_idx}')
    #随便展示一张
    # plt.imshow(train_data.data[37,:,:,:])
    # plt.show()
    #搭建神经网络
    # model = ImageNet()
    # summary(model,(batch_size,3,32,32))
    #模型训练
    # train_model(train_data)
    #模型测试
    evaluate_model(test_data)