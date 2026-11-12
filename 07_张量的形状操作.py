"""
涉及到的API：
    reshape():  不改变内容，按顺序排列
    unsqueeze():    在指定轴增加一个维度
    squeeze():      删除大小为1的纬度
    transpose():    #交换指定维度，一次只能交换两个
    permute():      #交换指定维度，一次能交换多个
    view():         #只能修改连续的张量的形状，所谓连续就是张量的显示顺序和张量在内存中的存储顺序一致（通常取决于创建的时候）
    conotiguous():  #把不连续的张量变成连续的张量，即基于张量显示的顺序，修改张量中存储的顺序
    is_contiguous():    #判断张量是否连续
"""
import torch

#reshape
a=torch.tensor([[1,2,3],[4,5,6]])
print(a.shape[0])   #shape第一个元素为列数

# b=a.reshape(3,2)
b=a.reshape(-1,1)
print(b)    #shape不改变内容

#unsqueeze
b=a.unsqueeze(0)    #(1,2,3)
c=a.unsqueeze(1)    #(2,1,3)
d=a.unsqueeze(2)    #(2,3,1)
print(b)
print(c)
print(d)

#squeeze
a1=torch.randint(1,50,size=(2,1,4,1,2,3,1))
print(a1.shape)
a2=a1.squeeze()
print(a2.shape)     #(2,4,2,3)

#transpose()
a2=torch.randint(1,50,size=(2,3,4))
a3=a2.transpose(0,1)   #转换第0维和第1维
print(a3.shape)          #(3,2,4)

#permute
a3=torch.randint(1,50,(2,3,4))
a4=a3.permute(2,1,0)        #重组后的张量维度索引对应原维度索引的2,1,0，即维度变为（4,3,2）
print(a4.shape)

# view()    只能修改连续张量的形状
t1=torch.randint(1,50,size=(2,3,4))
t2=t1.view(4,2,3)
print(t2.is_contiguous())
t2=t1.transpose(1,2)    #交换第1和第2个维度
print(t2.shape)                 #（2,4,3）
print(t2.is_contiguous())       #transpose()转完变为不连续
t3=t2.contiguous()              #t3变为连续
print(t3.is_contiguous())
t4=t3.view(2,3,4)
print(t4.shape)

