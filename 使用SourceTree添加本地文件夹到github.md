---
title: 使用SourceTree添加本地文件夹到github
tags: [SourceTree,gitHub]
categories: 技术修炼
---

### 1. 先在gitHub添加新的Repository，并复制url

### 2. 【可选】 [生成SSH Key 并添加到gitHub](https://blog.igevin.info/posts/generate-ssh-key-for-git/)

### 3. 在SourceTree中将已存在的文件夹添加到github 

1）在SourceTree中新建Repository，并找到对应文件夹
![](http://ww4.sinaimg.cn/large/006tNc79gy1fg753oag69j30qm0b6q52.jpg)

2）打开新建的Repository，点击Settings，设置远程Repository地址，将之前复制的url填入
![](http://ww4.sinaimg.cn/large/006tNc79gy1fg76dzn608j30lw0dcta3.jpg)

3）先pull一次，将远程仓库的Readme文件同步过来，然后commit下，就可以push了