---
title: python中的多线程与多进程
tags: [python,技术杂烩]
categories: 技术修炼

---

对于计算密集型任务来说，将cpu的效率发挥到100%是最理想的情况。而对于IO密集型任务来说则不需要占用太多CPU。

单个进程中可以同时启动多个线程，但多个线程同时共享同一个cpu核，所以对于计算密集型任务来说，使用多进程才能将cpu效率发挥至最高

参考如下两个例子启动多进程与多线程：

```python
# 多线程
from multiprocessing import pool
import os
import time
import numpy as np
def worker(id):
    print "worker"
    for i in range(1000000):
        np.sqrt(i**2)
    print str(os.getpid())+"\t"+str(id)
    print "end worker"

iters = range(10)
time1 = time.asctime()
pool = pool.ThreadPool(10)
num = pool.map(worker, iters)
pool.close()
pool.join()

print time1
print time.asctime()
```

```python
# 多进程
import multiprocessing
import time
import os
import numpy as np

def worker(id):
    print "worker"
    for i in range(1000000):
        np.sqrt(i**2)
    print str(os.getpid())+"\t"+str(id)
    print "end worker"

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=10)
    time1 = time.asctime()
    nb = range(10)
    pool.map(worker, nb)
    pool.close()
    pool.join() 

    print time1
    print time.asctime()
```

上面两个程序唯一的差别在于pool是用multiprocessing.pool.ThreadPool还是multiprocessing.Pool，用后者即实现了多进程。对比时间发现，后一个程序的运行时间是前一个时间的cpu数量分之一。

另外，进程启动数量并非越多越好，这取决于计算机的真正物理核数量，而不是虚拟线程数，获取方式为multiprocessing.cpu_count()/2。