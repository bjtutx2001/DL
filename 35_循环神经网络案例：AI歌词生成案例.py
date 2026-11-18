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
        self.num_chars=num_chars            #每次加载的数据条数
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

#测试
if __name__ == '__main__':
    unique_words,word_to_index,word_counts,corpus_index=bulid_vocab()
    # dataset=lyricsdataset(corpus_index,5)
    # print(len(dataset))
    data_load=data_processing(corpus_index)
    print(len(data_load))