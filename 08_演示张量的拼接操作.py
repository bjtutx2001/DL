"""
涉及到的API:
    cat()       #不改变维度，除了拼接的那个维度，其他维度必须保持一致
    stack()     #改变维度，两个张量维度必须保持一致，生成新的维度
"""
import torch

#cat()
a1=torch.randint(1,50,size=(2,3))
a2=torch.randint(1,50,size=(2,3))
t1=torch.cat((a1,a2),dim=1)
print(t1.shape)     #(2,3)+(2,3).dim(1)=(2,6)

t2=torch.cat((a1,a2),dim=-1)    #(2,6)
t3=torch.cat((a1,a2),dim=0)     #(4,3)

#stack()
a3=torch.randint(1,50,size=(2,3))
a4=torch.randint(1,50,size=(2,3))
b1=torch.stack([a3,a4],dim=0)   #(2,2,3)，新增0维度
b2=torch.stack([a3,a4],dim=1)   #(2,2,3)，新增1维度
b3=torch.stack([a3,a4],dim=2)   #(2,3,2)，新增2维度
