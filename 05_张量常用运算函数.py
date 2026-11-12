"""
常用的张量运算函数有：
    sum()
    max()
    min()
    mean()
    上面四个有dim参数，dim代表行或列，例如按行求和
    sqrt()      开方
    pow()       幂运算
    exp()
    log()       以e为底
    log2()
    log10()
"""
import torch
a=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(a.sum(dim=0))     #按照列求和
print(a.sum(dim=1))     #按照行求和
print(a.sum())          #整体求和

print(a.max(dim=0))     #按照列求最大值
print(a.max(dim=1))     #按照行求最大值
print(a.max())          #整体求最大值

print(a.pow(2))         #平方
print(a.pow(3))         #立方

print(a.sqrt())         #只能开平方，不能传参

print(a.exp())          #e的n次幂，n就是张量中的对应元素