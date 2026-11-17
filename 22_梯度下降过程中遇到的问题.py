"""
梯度下降算法优化过程中，可能会遇到以下几种情况：
    1、遇到平缓区，梯度值较小，参数优化变慢
    2、遇到鞍点，梯度为0，参数无法优化
    3、遇到局部极小值，参数不是最优
针对这些问题，有一些对梯度下降算法的优化方法：
    1、指数移动加权平均Exponentially Weighted Moving Average，EWMA：给最近的数据分配较大的权重，给较早的数据分配较小的权重，并且权重会随着时间呈指数形式衰减。
        假设有一组随时间变化的数据：x1，x2，x3......
        指数移动加权平均 的计算公式为：Vt=βVt'+（1-β）xt
        其中：Vt表示当前时刻的指数移动加权平均值
             Vt'表示上一时刻的指数移动加权平均值
             xt表示当前时刻的新数据
             β为衰减系数，范围为[0,1),常见取值为0.9,0.99
             一般情况下初始值V0设置为0
        这个公式可以理解成：新的平均值 = 历史平均值 × β + 当前数据 × (1−β)。
        它与普通平均值最大的区别在于：普通算术平均对所有数据赋予相同权重，而 EWMA 更重视最近的数据。
        EWMA 更重视最近的数据，是相对于更早的单个数据而言，而不是说当前数据比所有历史数据加起来还重要。
    2、动量法Momentum：结合当前梯度和历史梯度更新参数，基于指数移动加权平均
        当前梯度公式：Vt=βVt'+（1-β）gt
            Vt：当前梯度的指数移动加权平均
            β：动量系数，常取0.9
            gt：当前梯度
            Vt'：之前累计的梯度信息
        更新参数：θt=θt'-αVt
    3、自适应梯度算法AdaGrad（Adaptive Gradient Algorithm）:根据每个参数历史梯度的平方累积值，自动调整该参数的学习率。
        历史梯度较大的参数：学习率逐渐减小。
        历史梯度较小的参数：相对保留较大的学习率。
        不同参数拥有不同的有效学习率，而不是所有参数始终使用相同 的学习率。
        AdaGrad 引入了一个变量Gt，用于保存历史梯度平方的累积值。
            Gt=Gt'+gt^2
            初始值G0=0
        自适应学习率ηt=η/(sqrt((Gt)+e(epsilon)))
            η：基础学习率
            Gt：截止到当前迭代的梯度平方累计值
            e：防止分母为0的极小正数
        更新参数：θt=θt'-ηt*gt
        缺点：可能会使得学习率过早，过量的降低，导致模型后期学习率太小，很难找到最优解
    4、均方根传播算法RMSprop（Root Mean Square Propagation）：主要用于解决 AdaGrad 在训练过程中学习率持续减小，可能导致模型后期更新过慢的问题。
        核心思想：不再像 AdaGrad 那样累积所有历史梯度的平方，而是利用指数移动加权平均（EWMA），计算近期梯度平方的平均值，再根据这个平均值调整各参数的学习率。
        梯度平方和计算改成：Gt=βGt'+(1-β)gt^2
        自适应学习率ηt=η/(sqrt((Gt)+e(epsilon)))  PyTorch 默认的 RMSprop 使用的分母形式是sqrt((Gt)+e(epsilon)，放在根号外
        更新参数：θt=θt'-ηt*gt
    5、自适应矩估计Adam（Adaptive Moment Estimation），结合了Momentum和RMSprop的思想，既考虑历史梯度的变化方向，又根据梯度的大小动态调整每个参数的更新步长。
        梯度的更新采用Momentum：Vt=βVt'+（1-β）gt，表示 一阶 梯度更新
        学习率的更新采用RMSprop：Gt=βGt'+(1-β)gt^2，表示二阶梯度更新
                              真实学习率=η/(sqrt((Gt)+e(epsilon)))
        总体更新公式：W=W'-真实学习率*Vt
        Adam在初始化时，会另Vt和Gt为0，因此训练初期计算得到的移动平均值会偏向0，为了解决这一问题，引入偏差修正
                真实的梯度的一阶矩和二阶矩为：
                Vt_=Vt/(1-β1^t)      Gt_=Gt/(1-β2^t)，其中β就是计算一阶矩和二阶矩的衰减系数
总结：
    简单任务和较小模型用：SGD和Momentum
    复杂任务或者大量数据：Adam
    需要处理稀疏数据或文本数据：Adagrad或RMSprop


"""
#导包
import torch
import torch.nn as nn

#演示动量法
def dm01_momentum():
    #初始化权重参数
    w=torch.tensor([1,2],requires_grad=True,dtype=torch.float)
    #损失函数
    criterion=(w**2).sum()      #标量张量才能自动微分，调用backward(),有两个权重，调用sum()就得到了y=w1^2+w2^2
    # print(criterion)
    #创建优化器对象
    #参数1：要优化的参数，参数2：学习率，参数3：动量参数
    optimizer=torch.optim.SGD([w],lr=0.01,momentum=0.9)     #实际上，SGD就是momentum=0的动量法，即只考虑本次梯度
    #计算梯度，依次执行：梯度清零，反向传播，更新参数
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    #执行第二次迭代
    criterion=(w**2).sum()
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

#演示自适应学习率AdaGrad
def dm02_adagrad():
    w=torch.tensor([1,2],requires_grad=True,dtype=torch.float)
    criterion=(w**2).sum()
    #创建Adagrad优化器
    optimizer=torch.optim.Adagrad([w],lr=0.01)
    #清空梯度+反向传播+更新参数
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    #重复上述
    criterion=(w**2).sum()
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

#演示均方根传播算法RMSprop
def dm03_rmsprop():
    w=torch.tensor([1,2],requires_grad=True,dtype=torch.float)
    criterion=(w**2).sum()
    #定义优化器
    optimizer=torch.optim.RMSprop([w],lr=0.01)
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    #重复步骤
    criterion=(w**2).sum()
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

#演示自适应矩估计Adam
def dm04_adam():
    w=torch.tensor([1,2],requires_grad=True,dtype=torch.float)
    criterion=(w**2).sum()
    #定义优化器
    optimizer=torch.optim.Adam([w],lr=0.01,betas=(0.9,0.999))   #betas为梯度一阶矩和二阶矩的更新系数
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    #重复步骤
    criterion=(w**2).sum()
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')




#测试
if __name__=='__main__':
    # dm01_momentum()
    # dm02_adagrad()
    # dm03_rmsprop()
    dm04_adam()