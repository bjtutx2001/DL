"""
案例：基于杰伦歌词来训练模型，用给定的起始词，结合长度，进行AI歌词生成

实现步骤：
    1、获取数据，进行分词，获取词表
    2、数据预处理，构建数据集
    3、搭建RNN神经网络
    4、训练模型
    5、预测模型
"""

#导包
import torch
import jieba
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
import torch.optim as optim
import time

#定义函数，获取数据，构建词表
def bulid_vocab():
    #定义变量，记录：去重后的所有词，每行文本分词结果
    unique_words,all_words=[],[]
    for line in open('./data/jaychou_lyrics.txt','r',encoding='utf-8'):
        line_words = jieba.lcut(line)
        # print(line_words)       #['想要', '有', '直升机', '\n']
        all_words.append(line_words)    #结果为列表套列表
        for word in line_words:
            if word not in unique_words:
                unique_words.append(word)
    #统计去重后词的数量
    word_counts=len(unique_words)
    # print(word_counts)
    #构建词表，字典形式，键：具体的词，值：词索引
    word_to_index={word:i for i,word in enumerate(unique_words)}
    # print(word_to_index)
    #将歌词文本用词表索引表示
    corpus_index=[]
    for words in all_words:
        new_words=words.copy()
        new_words.append(' ')
        # print(words)
        for word in new_words:
            corpus_index.append(word_to_index[word])
            # print(word)
    # print(corpus_index)

    #返回所有词，词表，词的数量，歌词切词结果对应的词索引，每句歌词结尾用单空格划分
    return unique_words,word_to_index,word_counts,corpus_index

#数据预处理
#RNN 歌词生成案例，不能把每个词看成互相独立的样本，因为预测下一个词时，需要考虑前面已经出现的词。
#采用滑动窗口（Sliding Window）的方法构建训练数据集。
#核心思想：用连续的一段词作为输入x，将这段词整体向后移动一个位置，作为标签y。这样，模型在每一个时间步都学习预测下一个词，而不是只预测一次。

# 方法1：定义函数实现
#参数为：无重复词库，词表，去重词数量，索引切词结果对应的词索引
def data_processing(corpus_index):
    # 每个训练样本的序列长度
    seq_len = 5
    #定义空列表，存储特征和预测
    x_data = []
    y_data = []

    # 滑动窗口构建样本
    for i in range(len(corpus_index) - seq_len):    #滑动次数等于歌词切词表长度减去每次训练的序列长度
        x = corpus_index[i:i + seq_len]             #每次取出seq_len长度的词作为标签
        y = corpus_index[i + 1:i + seq_len + 1]     #每次取出seq_len长度的词作为预测结果，相比于标签后移一位

        x_data.append(x)
        y_data.append(y)

    # 转换为张量
    x_data = torch.tensor(x_data, dtype=torch.long)
    y_data = torch.tensor(y_data, dtype=torch.long)

    # print(x_data)
    # print(y_data)

    # 构建数据集
    dataset = TensorDataset(x_data, y_data)

    # 构建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )
    return dataloader

#方法2：定义数据集类，继承自torch.utils.data.Dataset
class lyricsdataset(torch.utils.data.Dataset):
    def __init__(self,corpus_index,num_chars):
        self.corpus_index=corpus_index      #文档数据中词的索引
        self.num_chars=num_chars            #每个样本中包含多少词
        self.world_counts=len(self.corpus_index)    #文档数据中词的数量，不去重

    #魔法方法，操作len(obj)时会调用__len__方法，obj是类对象，不会自动计算元素数量
    def __len__(self):
        return max(0,len(self.corpus_index)-self.num_chars)     #0是为了防止返回负数
    #魔法方法，对类对象进行索引操作时obj[index]，会调用此方法
    def __getitem__(self, index):
        start = min(max(index,0),self.world_counts-self.num_chars-1)
        x=self.corpus_index[start:start+self.num_chars]
        y=self.corpus_index[start+1:start+self.num_chars+1]
        return torch.tensor(x),torch.tensor(y)

#搭建神经网络
class textgenerator(nn.Module):
    def __init__(self,unique_count):  #传入参数：唯一词表大小
        super().__init__()  #初始化父类成员
        self.unique_count=unique_count
        #词嵌入层,
        # 输入参数：唯一词表大小，输出参数：词向量维度
        self.embed=nn.Embedding(self.unique_count,128)
        #循环层,
        #输入参数：词向量维度，隐藏状态维度，rnn层数
        self.rnn=nn.RNN(128,256,1,batch_first=True)
        #全连接层，也是输出层
        #参数：输入特征大小，也就是隐藏状态维度；唯一词表大小，计算每个词的预测分数，分数最大的就是预测结果
        self.out=nn.Linear(256,unique_count)    #后续用多元交叉损失熵函数计算损失

    def forward(self,x,h0=None):
        #词嵌入
        x=self.embed(x)         #嵌入后x的格式为(batch,句子长度，词向量维度)
        #RNN前向传播，要求传入的格式为：（句子长度，batch，词向量维度），可以在创建RNN时指定batch_first=True
        output,hn=self.rnn(x,h0)        #output记录的是最后一层RNN在每一个时间步产生的隐藏状态，hn记录的是每一层RNN在最后一个时间步产生的隐藏状态
        #输出                             output用于计算预测结果                             hn用于后续继续计算隐藏状态
        # print(x.shape,output.shape,hn.shape)
        output=self.out(output)

        return output,hn

#模型训练
#传入构建词典时生成的数据
def train_model(unique_words,word_to_index,word_counts,corpus_index):
    #获取数据集
    dataset=lyricsdataset(corpus_index,16)       #每个样本16个词
    #初始化神经网络模型并移动带GPU
    device=torch.device('cuda:0')
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=textgenerator(word_counts).to(device)
    # model=textgenerator(word_counts)
    model.train()
    #创建数据加载器对象
    # dataloader=DataLoader(dataset,batch_size=5,shuffle=True)
    dataloader=DataLoader(dataset,batch_size=5,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True)    #每批5个句子
    # 定义损失函数
    criterion=nn.CrossEntropyLoss()
    #定义优化器
    optimizer=optim.Adam(model.parameters(),lr=0.0001)
    #定义变量，训练轮数
    epochs=30
    for epoch in range(epochs):
        #定义训练开始时间，迭代次数，训练总损失
        start_time,iter_num,total_loss=time.time(),0,0.0
        #循环训练
        for x,y in dataloader:  #数据加载器对象会在内部完成索引操作，实现数据调用并返回
            #数据移到GPU
            x=x.to(device,non_blocking=True)
            y=y.to(device,non_blocking=True)
            #预测
            y_pred,hn=model(x)
            #计算损失，多元交叉损失熵要求类别维度必须位于索引1的位置（二维输入则是最后一维）
            y_pred=y_pred.permute(0,2,1)
            loss=criterion(y_pred,y)
            total_loss+=loss.item()
            #梯度清零+反向传播+更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num+=1

        #打印本轮训练信息
        print(f'轮数：{epoch+1},平均损失：{total_loss/iter_num}，耗时：{time.time()-start_time:.2f}')
        #保存模型
    torch.save(model.state_dict(),'./model/singer_model.pth')
    print(f'保存成功！')

#模型测试
def evaluate_model(start_word,setence_length,unique_words,word_to_index,word_counts):
    #创建模型
    model=textgenerator(word_counts)
    #加载模型参数
    model.load_state_dict(torch.load('./model/singer_model.pth'))
    #将输出的初始值转成索引
    word_index=word_to_index[start_word]
    #定义列表，存放产生的词的索引
    generate_index=[word_index]
    for i in range (setence_length):
        model.eval()
        #预测
        #思路，先根据初始词预测下一个句子，然后把完整的预测结果（初始+预测）作为输入，预测完整的句子，取预测结果的最后一个词，填入到索引列表，再继续用完整的预测结果作为输入预测
        output, hn=model(torch.tensor(generate_index,dtype=torch.long).unsqueeze(dim=0))    #输入要是二维的
        #获取预测结果
        # 理解方法：指定哪个dim，就让哪个维度的索引发生变化，其他维度的索引保持不变，然后比较这些位置上的元素。
        # 例如dim = 0，就是比较x[0, i, j]和x[1, i, j]；dim = 1，就是比较x[k, 0, j]和x[k, 1, j]。
        # word_index=torch.argmax(output)   #不写dim，会把output展平成一维张量，找到最大元素对应的索引
        #argmax输出的是张量
        word_index=torch.argmax(output[0,-1,:]).item()      #获取最后一个时间步 预测的下一个词
        # print(word_index)
        generate_index.append(word_index)                   #一个词一个词的添加

    #将所索引转成词
    for idx in generate_index:
        print(f'{unique_words[idx]}',end='')



#测试
if __name__ == '__main__':
    #获取数据
    unique_words,word_to_index,word_counts,corpus_index=bulid_vocab()
    # dataset=lyricsdataset(corpus_index,5)
    # print(len(dataset))
    #创建数据加载器
    # 方法1
    # data_load_1=data_processing(corpus_index)
    # print(len(data_load))
    # for x,y in data_load:
    #     print(x,y)      #结果为一个张量，代表每个句子拆出的词在词表的索引
    #方法2
    # data_set=lyricsdataset(corpus_index,16)
    # data_load_2=DataLoader(data_set,batch_size=5,shuffle=True)
    # for x,y in data_load_2:
    #     print(x,x.shape,y,y.shape)  #结果为一个张量，维度为批次数*每个样本词数

    #创建模型对象
    # singer=textgenerator(word_counts)
    #查看模型参数维度
    # for name,param in singer.named_parameters():
    #     print(f'name: {name}, param: {param.size()}')

    #模型训练
    # train_model(unique_words,word_to_index,word_counts,corpus_index)

    #模型测试
    evaluate_model('安静',20,unique_words,word_to_index,word_counts)