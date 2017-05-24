---
title: Ubuntu安装NVIDIA显卡驱动及tensorflow-gpu
tags: [深度学习,GPU,tensorflow]
categories: 技术修炼
---

一直以来想试试用显卡做深度学习计算时什么感觉，刚开始学习神经网络的时候在AWS上开启过（当时开启显卡计算忘了关，一个多星期被收了1000多软妹币，好在AWS客服比较善良，沟通后直接把钱还给我了:smirk:）；中间有一次因也无需要在阿里云开了个GPU计算，那叫一个与时间赛跑，搭建环境分分钟都是钱:joy:；这次终于花重金，400大洋:thumbsdown::thumbsdown:自己买了个750ti（原来的显卡是HD6850网游戏那是杠杠滴，这也是一直没换的原因），虽然牌子不咋地，但是一点就亮。

### 1. 安装显卡驱动

安装显卡驱动可能是最麻烦的，各种毛病。当然，如果不确定会不会遇到这些问题，那就直接装，看遇到什么问题在逐个排除。

**nouveau**加入黑名单
sudo vim /etc/modprobe.d/blacklist.conf
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist rivatv
blacklist nvidiafb

**关闭图形界面**
sudo service lightdm stop
sudo service gdm stop (或者是gdm3)

然后sudo sh NVIDIA-….-xxx.sh安装

### 2. 安装Cuda tools, cuDNN

Cuda默认带着显卡驱动，上一步装过显卡驱动，这里显卡驱动可以跳过。
cuDNN的安装过程：
tar -zxvf cudnn-8.0-linux-x64-v5.0-ga.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

### 3. tensorflow

在tensorflow加入了pip安装过程后，tensorflow的安装变得极其简单。pip的安装方式可能会导致程序运行时提示从"SSE, AVX,FMA"编译可以加快cpu的运行速度，不过这个也无伤大雅。在前面两步执行成功以后，下面的命令安装的tensorflow可以 成功将gpu用上。750ti虽然不咋地，但比起本地cpu的运行速度，还是快了7倍以上。
pip install tensorflow-gpu