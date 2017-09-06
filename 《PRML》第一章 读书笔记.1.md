---
title: 《PRML》第一章 读书笔记.1
tags: [机器学习,PRML,模式识别]
categories: 机器学习
---
[TOC]

## 模式识别概述

### 1、模式识别
使用算法自动发现数据中的规律，并利用这些规律实现回归、分类等任务。例如**手写数字识别**（MINIST数据）![](https://ws3.sinaimg.cn/large/006tKfTcgy1fgp3424nf1j30pa0aq75d.jpg)

### 2、模式识别的几个步骤

* 特征工程（数据准备）：
  ①一般同一个算法需要的输入数据都是一致的 
  ②有助于提高计算速度、收敛速度（如归一化）![Fig.2](https://ws3.sinaimg.cn/large/006tKfTcgy1fgp3dfjzrvj30gq05m754.jpg "Fig.2")

* 模型（算法）设计：线性回归、支持向量机（SVM）、神经网络

* 训练与验证：
  1) 将数据按比例（$\eta=0.8$)将数据分为训练集与测试集；
  2) 训练阶段，不断调整参数，以期模型能正确判断
  3) 验证：通过验证模型在新的样本上的正确性一测试其泛化能力；如果训练阶段表现非常好、验证阶段表现太差则成为过拟合。（对过去了如指掌，对未来一无所知。 —Luke）

### 3、三个任务

* 监督学习：从银行卡号识别别所属银行、对应卡产品，再比如手写数字识别、画出图片中的卡边界（Fig.2)

* 非监督学习：[鸡尾酒会问题](http://www.endolith.com/wordpress/2009/11/22/a-simple-fastica-example/)（使用独立成分分析）

* 增强学习：通过反馈评分的方式训练机器人行走、搬东西、玩游戏等

这三个任务所处理的场景各不相同，但其中的一些基本概念、思想是一致的。

### 4、多项式拟合

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fgp606utlnj30kq0fc74x.jpg" style="zoom:50%">

+ 数据形式（x, y)

+ 模型（算法）：多项式拟合![](https://ws4.sinaimg.cn/large/006tNc79gy1fgp65s70puj30pa03iaae.jpg)

+ 参数学习：最小化损失函数![](https://ws1.sinaimg.cn/large/006tNc79gy1fgp6bkq09nj30eq04gaa9.jpg)

+ 模式选择：选择合适的M，确定模型![](https://ws4.sinaimg.cn/large/006tNc79gy1fgp6iwhkk4j31640vwgps.jpg)

  *trick: 使用均方根对比不同size的数据下模型的损失函数![](https://ws3.sinaimg.cn/large/006tNc79gy1fgp806gv37j30v40dkq4a.jpg)*

+ 如果把数据增加，对于M=9的模式![](https://ws4.sinaimg.cn/large/006tNc79gy1fgp6we4m86j31720fsq5x.jpg)

  *可以看到，当数据增加时，原本过拟合的模型也可以拟合的比较好，也就是说<u>数据量越大越有利于复杂的模型拟合数据</u>，一般认为数据量应该是参数量的5~10倍。后面将看到，参数数量并不是最合适的表示模型复杂度的参数。*

+ 实际情况是数据量不多，模型复杂容易过拟合。防止过拟合：正则化、贝叶斯方法

  岭回归：![](https://ws2.sinaimg.cn/large/006tNc79gy1fgp7rc3lzfj30jk04a74l.jpg)![](https://ws1.sinaimg.cn/large/006tNc79gy1fgp83bwwnwj30uq0r4mzf.jpg)


  使用正则化以后，对于合适的正则化，复杂的模型也可以较好的拟合数据，并不会出现过拟合。而当正则化项过大时，也会出现欠拟合的情况。

## 概率论

模式识别中的一个关键概念：不确定性。造成不确定性的因素是噪声及有限的数据。结合决策理论，即便有用的信息不完整或者模糊不清，也可以做出相对最优的预测。

### 1、基本概念

<img src="https://ws4.sinaimg.cn/large/006tNc79gy1fgpdbo5eimj31cs0xgjw1.jpg" style="zoom:40%">

+ 联合概率：$p(X=x_{i},Y=y_{j})=\frac{n_{ij}}{N}$
+ 边缘概率：$p(X=x_{i})=\sum_{j=1}^{2}p(X=x_{i},Y=y_{j})$
+ 条件概率：$p(Y=y_j|X=x_i)=\frac{n_{ij}}{n_i}$

边缘概率的另一种表示：$p(X=x_i)=\frac{n_i}{N}$
所以得到：
$$
p(X=x_i,Y=y_j)=\frac{n_{ij}}{N}
=\frac{n_{ij}}{n_i}·\frac{n_i}{N}
=p(Y=y_j|X=x_i)p(X=x_i)
$$
独立变量：如果两个变量的联合概率可以分解为边缘概率的乘积，那么这两个变量独立。$P(a,b) = p(a)p(b)$

> 思考： 上面图中的X，Y是否独立？

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fgpfun2ds4j30vu0nmwgo.jpg" style="zoom:40%">

上面这幅图表示两幅图的颜色直方图分布，其中考虑了周期性。各个颜色值所对应的概率分布通过简单的数据统计与总数值相除得出。
$$
p(h)=\frac{n_h}{\sum_{i=0}^{255}n_i}
$$

### 2、 概率密度

对于连续变量，概率一般表示为概率密度$p(x)$,

<img src="https://ws2.sinaimg.cn/large/006tNc79gy1fgpey6ewpdj30ou0iq0tx.jpg" style="zoom:50%">

已知概率密度为$p(x)$, 则x在区间(a,b)范围内的概率为$p(x\in(a,b))=\int_a^bp(x)dx$

概率密度的性质：
$$
p(x) \ge 0
$$
$$
\int_{-\infty}^{\infty}p(x)dx = 1
$$

累积概率函数（cdf）:
$$
P(x)=\int_{-\infty}^{x}p(x)dx
$$

连续变量的边缘概率分布与联合概率分布：

<img src="https://ws1.sinaimg.cn/large/006tNc79gy1fgpguqbif7j30gs05k74n.jpg" style="zoom:50">

### 3、 期望与协方差(以离散变量为例)

期望：$E[f]=\sum_{x}p(x)f(x)$
​	    $E[f]=\frac{1}{N}\sum f(x_n)$

方差：$var[f]=E[(f(x)-E[f(x)])^2]$
​	    $var[f]=E[f^2]-E[f]^2$

协方差: $cov[x,y]=E_{x,y}[\{x-E[x]\}\{y-E[y]\}]$

### 4、 贝叶斯公式

$$
p(x,y)=p(x|y)p(y)=p(y|x)p(x)
$$

$$
\Longrightarrow p(y|x)=\frac{p(x|y)p(y)}{p(x)}
$$

> 考虑一个医疗诊断问题，有两种可能的假设：（1）病人有癌症。（2）病人无癌症。样本数据来自某化验测试，它也有两种可能的结果：阳性和阴性。假设我们已经有先验知识：在所有人口中只有0.008的人患病。此外，化验测试对有病的患者有98%的可能返回阳性结果，对无病患者有97%的可能返回阴性结果。假设现在有一个新病人，化验测试返回阳性，是否将病人断定为有癌症呢？
> $$
> P(癌症|阳性) = \frac{P(阳性|癌症)P(癌症)}{P(阳性)}
> $$
> $P(阳性|癌症)=0.98，P(癌症)=0.008，P(阳性)=0.008*0.98+0.992*0.03$$P(癌症|阳性)=20.85\%$

贝叶斯理论的其他应用：

<img src="https://ws2.sinaimg.cn/large/006tNc79gy1fgqqiuqfecj30gy02274i.jpg" style="zoom:50%">

> 先验概率：是指根据以往经验和分析得到的概率
> 后验概率：指在得到“结果”的信息后重新修正的概率

### 5、高斯分布

![](https://ws2.sinaimg.cn/large/006tNc79gy1fgqqt10b9kj30qe04amxn.jpg)

![](https://ws2.sinaimg.cn/large/006tNc79gy1fgqqtkx7c7j30qy0jy758.jpg)

中心极限定律：大量相互独立的随机变量，其均值（或者和）的分布以正态分布为极限（采样次数趋向无穷大的时候，就越接近正态分布）。

> 比如随机间隔时间，从样本无限大样本T中以100个为基数求平均间隔时间，最后会发现平均间隔时间服从正态分布。
>
> 或者掷骰子，每个样本以10次出现的点数求平均值，当样本趋向于无穷时，平均值的分布为正态分布。

### 6、 最大似然估计与最大后验概率估计

**最大似然估计**：模型已定，参数未知

<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fgp606utlnj30kq0fc74x.jpg" style="zoom:50%">

> 考虑前面的多项式拟合：给定参数$\omega,\beta$后, 对于数据点x的取值t，相对于理论值$y(x,\omega)$应该服从以理论值为重心的高斯分布。
> <img src="https://ws3.sinaimg.cn/large/006tNc79gy1fgrgch19g3j30ik02e3yr.jpg" style="zoom:50%">
> ☆从上式进行预测时不再是点到点的形式，而是给出t的概率分布形式。

所以：对于全体数据，$\pmb{x} = (x_1,x_2,…,x_n)^T,\pmb{t}=(t_1,t_2,…,t_n)^T$, 整体存在的概率(因各数据点*相互独立*，所以用乘法法则），也就是**<u>似然函数</u>**：

<img src="https://ws1.sinaimg.cn/large/006tNc79gy1fgri231mi7j30ly04oq3c.jpg" style="zoom:50%">

> 思考：为何此处相互独立，而前面x，y不独立。

上面的似然函数不好直接求解，一般转化为对数似然函数：
![](https://ws1.sinaimg.cn/large/006tNc79gy1fgriabflm4j30yy04awf5.jpg)


上式对w求导，得：
<img src="https://ws3.sinaimg.cn/large/006tNc79gy1fgric4xa79j30jo052jrp.jpg" style="zoom:50%">

**最大后验估计**：在先验分布的前提下获得的参数估计
对于所求的w，假设预先知道其分布形式为$p(w)$, 则：
$$
p(w|x,t,\beta)=\frac{p(t|x,w,\beta)p(x,\beta)p(w)}{p(x,t,\beta)}\propto p(t|x,w,\beta)p(w)
$$
即：最大后验估计正比于最大似然估计与先验估计的乘积。