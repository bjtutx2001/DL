"""
词嵌入层作用：
    把词或词对应的索引转换成词向量
步骤：
    1、切词，可以使用jieba分词器
    2、把词变成词索引，可以自己创建词表，也可以参考其他词表，词表中能找到索引对应的词
    3、创建词向量，一般情况下，词向量的初始化都是随机的，这些随机数最开始并不代表真实的词语语义。为什么不采用全0：让不同的词拥有了不同的初始表示

案例：演示词嵌入层

"""
#导包
import torch
import jieba
import torch.nn as nn

#创建一句话，并分词
text='我今天晚上吃了辣椒炒肉盖饭'
words=jieba.lcut(text)
print(words,len(words))

#创建词嵌入层
# Embedding 的中文含义是“嵌入”，在机器学习和深度学习中，通常指将某个对象映射到一个向量空间中，用向量来表示这个对象。
#参数1：词个数，参数2：词向量维度
embed=nn.Embedding(len(words),4)

#获取每个词对应的下标
#enumerate会返回列表中每个元素及其对应的下标（索引）
for i,word in enumerate(words):
    # print(i,word)
    #把词索引（张量形式）转成词向量，构建词索引和词向量的对应关系，而不是词本身和词向量的对应关系
    word_vector=embed(torch.tensor(i))
    print(f'词语：{word},词向量：{word_vector}')