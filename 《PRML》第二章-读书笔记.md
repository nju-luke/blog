---
title: 《PRML》第二章 读书笔记.1 
tags: [机器学习、PRML、模式识别]
categories: 机器学习
---
[TOC]

> 频率学派：选择特定的参数来获得合适的分布，比如通过似然函数。贝叶斯学派：先估计先验概率，再结合数据计算后验概率。
>
> 参数法：对分布假设一个特定的函数形式，缺点是对某些具体的场景不适用。
> 非参数法：分布形式主要依赖于数据集。仍然包含参数，但这些参数控制模型负责度而不是分布形式。

## 2.1 二元分布

考虑二元随机变量$x\in \{0,1\}$, x=1的概率用$\mu$表示 $p(x=1|\mu)=\mu$, 其中$0 \leq \mu \leq 1$, x=0的概率为 $p(x=0|\mu)=1-\mu$。x的分布则可以写为：
$$
Bern(x|\mu)=\mu^x(1-u)^{1-x}
$$
上式就是伯努利(Bernoulli)分布。容易验证：
$$
E[x]=\mu \\
var[x]=\mu(1-\mu)
$$
假设有数据集$D=\{x_1,…,x_N\}$，根据数据是从$p(x|\mu)$独立抽样得到可构建似然函数:
$$
P(D|\mu)=\prod_{n=1}^N p(x_n|\mu)
$$
对于频率学派来说， 可以通过最大化似然函数，或者最大化对数似然函数来估计$\mu$。在N次试验中，正面出现的次数为m，则容易求出$\mu=\frac{m}{N}$。假设丢三次硬币，3次的结果都是正面朝上，则$\mu=1$, 也就是说最大似然将预测未来的结果都为正面，但常识告诉我们这是有问题的，另外这也是一个过拟合的问题。如果结合对$\mu$的先验知识，该如何判断呢？

### 二项分布

在N次试验中观察到x=1的次数m的分布为二项(Binomial分布：
$$
Bin(m|N,\mu)=C_N^m \mu^m(1-\mu)^{N-m}
$$

$$
E[m]=\sum_{m=1}^NmBin(m|N,\mu)=N\mu \\
var[m]=N\mu(1-\mu)
$$

### 2.1.1 $\beta$ 分布

最大后验：$posterior \propto likelihod * prior$
二项分布的似然函数是$\mu^x(1-\mu)^{1-x}$，如果我们选择先验分布正比于$\mu$和$(1-\mu)$的幂次，那么后验分布将和先验分布具有相同的形式，该性质称为共轭性。

二项分布的共轭分布为：
$$
Beta(\mu|a,b)=\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}
$$
其中$\Gamma$表示伽马函数。a、b取不同值时的伽马分布：

![](https://ws4.sinaimg.cn/large/006tNc79gy1fhtma77b7nj315y0tg410.jpg)

如果在N次试验中出现m次正面、$l$次反面，则后验分布:
$$
p(\mu|m,l,a,b)\propto \mu^{m+a-1}(1-\mu)^{l+b-1}
$$
这是另一种形式的Beta分布，对比Beta分布公式可标准化系数：
$$
p(\mu|m,l,a,b) =\frac{\Gamma(m+l+a+b)}{\Gamma(m+a)\Gamma(l+b)}\mu^{m+a-1}(1-\mu)^{l+b-1}
$$
在给定数据D下，x的分布：
$$
p(x=1|D) = \int_0^1p(x=1|\mu)p(\mu|D)d\mu = \int_0^1\mu p(\mu|D)d\mu = \mathbb{E}[\mu|D]
$$

对于后验分布，使用Beta分布的期望公式，得到：
$$
p(x=1|D) = \frac{m+a}{m+a+l+b}
$$

从上式可以看出，当$m,l \to \infty$即$N\to\infty$时，上式退化为最大似然$\frac{m}{N}$。

## 2.2 多元变量

将二元变量分布的范围扩大到K个取值，取每一个值的概率分布为$\mu = (\mu_1,...,\mu_K)^T$,那么x的分布为：
$$
p(x|\mu) = \prod\limits_{k=1}^K\mu_k^{x_k}
$$
那么，有N个独立观测值的数据D的似然函数为：
$$
p(D|\mu) = \prod\limits_{n=1}^N\prod\limits_{k=1}^K\mu_k^{x_{nk}} = \prod\limits_{k=1}^K\mu_k^{(\sum_nx_{nk})} = \prod\limits_{k=1}^K\mu_k^{m_k}
$$

其中$m_k$可理解为观察到x取值为k的次数。再由$\mu$的标准化条件即可通过拉格朗日乘数法对似然函数求解：
$$
\sum\limits_{k=1}^{K}m_k\ln\mu_k + \lambda(\sum\limits_{k=1}^K\mu_k - 1 ) \\
\mu_k=-m_k/\lambda
$$
代入限制条件$\sum_k\mu_k=1$得$\lambda=-N$。得到最大似然解：$\mu_k^{ML} = \frac{m_k}{N}$。

在参数$\mu$，及总观测数N的条件下，对应观测值$m_1,m_2,...,m_K$的联合分布：
$$
Mult(m_1,...,m_k|\mu,N) = \binom{N}{m_1m_2...m_k}\prod\limits_{k=1}^K\mu_k^{m_k}
$$

### 2.2.1 The Dirichlet Distribution

参考二元分布，多项分布的共轭分布为:
$$
p(\mu|\alpha) \propto \prod\limits_{k=1}^{K}\mu_k^{\alpha_k - 1}
$$
其标准形式，狄利克雷分布：
$$
Dir(\mu|\alpha) = \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1)...\Gamma(\alpha_K)}\prod\limits_{k=1}^K\mu_k^{\alpha_k - 1}\\
其中 \alpha_0 = \sum\limits_{k=1}^K\alpha_k
$$

## 2.3 高斯分布














