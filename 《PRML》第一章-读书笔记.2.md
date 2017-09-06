---
title: 《PRML》第一章 读书笔记.2
tags: [机器学习、PRML、模式识别]
categories: 机器学习
---
[TOC]

##  模式选择

回顾前面的多项式拟合，多项式的阶数决定了模型的复杂度。另外，正则系数$\lambda$ 的大小限制了模型的复杂度。那么什么样的模型是最好的模型呢？等价于确定每一个超参数的值。

<img src="https://ws4.sinaimg.cn/large/006tNc79gy1fgp65s70puj30pa03iaae.jpg" style="zoom:50%">

<img src="https://ws2.sinaimg.cn/large/006tNc79gy1fgp7rc3lzfj30jk04a74l.jpg" style="zoom:50%">

### 1. 交叉验证

将数据按比例分为S（下图中S=4）份，每次训练使用其中的一份作为验证集，其余作为训练集。对每一个模型$M_i$进行4次训练，得到S个错误率， 对S各错误率求平均值即得到模型的综合错误率$\eta_i$。

<img src="https://ws4.sinaimg.cn/large/006tNc79gy1fh998qm3r0j30kw0c0dgm.jpg" style="zoom:50%">

对可变参数的n个模型，执行上述训练，选n个模型中错误率最小的模型作为最终模型。

### 2.信息量的判别

增加参数可使得似然概率增大，但是却引入了额外的变量。引入额外变量是的模型过于复杂。AIC和BIC都在目标式中添加了模型参数个数的惩罚项。

AIC：Akaike information criterion
$$
\ln p(D|W_{ML})-M
$$

在损失函数中加上参数个数的惩罚项。其中前半部分表示拟合最佳时的对数似然，M表示可训练参数数量。





## 维度诅咒

> 低维不可分的问题，映射到高维以后就可以区分！！！

![](http://img.my.csdn.net/uploads/201304/03/1364952814_3505.gif)

例子：如何给图中x分类（红绿蓝）。（原始数据为十维，图中画出其中两维）

<img src="https://ws4.sinaimg.cn/large/006tKfTcgy1fhr7vv26kdj30qw0q0dpy.jpg" style="zoom:40%">

简单的方法是将数据分块，数据点落在的块中哪一个类别的数据最多，分为哪一类。（类似Knn：找到目标距离最近的k个样本，取k个样本中类别最多的）

<img src="https://ws3.sinaimg.cn/large/006tKfTcgy1fhr7y9hxe0j30q20pywnq.jpg" style="zoom:40%">

**随着维度的增加，分块的数量呈指数被增加！！**但事情况是，无法找到如此多的训练数据填到每一个分块中。

<img src="https://ws1.sinaimg.cn/large/006tKfTcgy1fhyfdcfp5yj30yg0fk40a.jpg" style="zoom:40%">

### 球体积计算

二维：$V=\pi r^2$

三维：$V=\frac{4}{3} \pi r^3$

D维：$V=K r^D$

D维下球壳体积所占整个球体积的比例：
$$
\frac{V(r)-V(r-\epsilon)}{V(r)}=1-(1-\frac{\epsilon}{r})^D
$$
取r=1，对上式作图：

<img src="https://ws2.sinaimg.cn/large/006tKfTcgy1fhr8qsnjznj30ri0r8q5q.jpg" style="zoom:50%">

从图中可以看出：随着维度增加，球的体积逐渐聚集到球壳上。

原本区分并不明显的样本，由于维度的增加，其在特定维度上的特征也变得更加清晰。

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fhr9jbmmwxj30tw0t60wp.jpg" style="zoom:50%">

> 所以，随着维度的增加，原本不可区分的样本，因为其特征在新的维度上比较明显，故投影到高维即可实现分类。



## Decision Theory

1. 最小化误分率

2. 使用带权重的损失函数（最小化期望损失）

   > 考虑癌症诊断中的两个问题的代价：
   > ①把患者诊断为健康
   > ②把健康人诊断为幻癌症

   ![](https://ws1.sinaimg.cn/large/006tNc79gy1fhra8zgboxj308r03274d.jpg)

3. 设置拒绝条件: 两条线的和为1，当未超过阈值$\theta$时，始终拒绝。
   <img src="https://ws4.sinaimg.cn/large/006tNc79gy1fhrajrey7dj30r20kgtaa.jpg"  style="zoom:50%">






## Information Theory （信息论）

信息量（多少）的定义：可看做对x值的“惊喜程度”。确定的事：0；比较确定的事：较少的信息量；很不确定的事：较多的信息量。

熵（Entropy）：有两个独立变量x、y，观察到两个变量获得的信息为h(x)、h(y)，则整体信息应为两个变量获得信息的和$ h(x,y) = h(x) + h(y)$。另外，对于独立变量，联合概率$p(x,y)=p(x)p(y)$。根据这两个条件，h的形式应为：
$$
h(x)=-log_2 p(x) \\
H[x]=-\sum_x p(x)log_2 p(x)
$$
![](https://ws2.sinaimg.cn/large/006tNc79gy1fhrljzunt8j30tz0cawff.jpg)

熵的应用：最大熵原理。
假设有100块钱放在下面两个盒子中，那么在黄色盒子中的概率是多少？

<img src="https://ws2.sinaimg.cn/large/006tNc79gy1fhrly7jgf9j30ze0hwt9e.jpg" style="zoom:50%">
在其中一个盒子的概率与整体熵的关系：

<img src="https://ws2.sinaimg.cn/large/006tNc79gy1fhrm5kv9aij30ug0ne75k.jpg" style="zoom:50%">

按照常识，在没有任何信息的前提下，我们一定会猜测在两个盒子中的概率都为0.5。当概率相等时，熵正好达到最大。

其他应用：词性标注、短语识别、指代消解、语法分析、机器翻译、文本分类、问题回答、语言模型。



### 条件熵与互信息

如果两个变量相互独立，则其联合概率等于其边缘概率的乘积；否则，可通过其联合概率与边缘概率乘积来判断他它们的分布是否接近。

![](https://ws4.sinaimg.cn/large/006tNc79gy1fhrnig0sruj311m084t9x.jpg)![](https://ws4.sinaimg.cn/large/006tNc79gy1fhrnlawd5cj30ps054gm3.jpg)
![](https://ws2.sinaimg.cn/large/006tNc79gy1fhrnm3g7alj30uk072t9t.jpg)
![](https://ws2.sinaimg.cn/large/006tNc79gy1fhrnn7mvwkj30pe02m0t3.jpg)



