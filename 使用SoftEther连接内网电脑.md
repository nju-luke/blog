---
title: 使用SoftEther连接内网电脑
tags: [vpn]
categories: 技术修炼
---

使用SoftEther可实现到外网到公司内网的vpn连接，步骤如下：

1. 参考[softether server的安装配置（ubuntu、debian）](http://www.lichanglin.cn/softether%20server%E7%9A%84%E5%AE%89%E8%A3%85%E9%85%8D%E7%BD%AE%EF%BC%88ubuntu%E3%80%81debian%EF%BC%89/)，先在长期运行的服务器端安装softEther，服务端安装好softEther后请参考下一步

2. 需从外网访问最简单的方法可能是通过配置[VPN Azure](http://www.softether.org/4-docs/2-howto/6.VPN_Server_Behind_NAT_or_Firewall/2.VPN_Azure)，也许端口也可以实现（暂未尝试）。配置Azure，直接从[官网下载](http://www.softether-download.com/en.aspx?product=softether)Sever Manage，windows的比较好用。按照http://www.vpnazure.net/en/的左侧配置即可

3. 客户端实现连接可参考右侧。我的windows是通过sstp的方式连上的，在mac下l2tp尝试了半天连不通

另外，服务器启动SoftEther后会变得很卡，至少对ssh远程连接是有很大影响的。