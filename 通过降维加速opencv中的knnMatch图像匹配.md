---
title: 通过降维加速opencv中的knnMatch图像匹配
tags: [PCA, opencv, sift]
categories: 图像处理
---
>问题描述：opencv中knnMatch是一种蛮力匹配，基本原理是将待匹配图片的sift等特征与目标图片中的全部sift特征一对n的全量便利，找出相似度最高的前k个。当待匹配的图片增多时，需要的计算量太大，所以考虑是否可以通过降维的方式减少计算过程中的时间花费。

## 1. PCA降维
从物理的角度来说，所有的数据都可以看作物体（object）在选定特征（坐标轴）上的投影，而这个数值的大小代表这个object在这个特征上的表现力（强度）。为了尽可能详细的去表达一个或者一类object，有时候选取的特征会过多，就造成了数据冗余，而这些冗余的数据对于计算来说花费是十分昂贵的。所以涌现了诸如PCA，LDA，FA等降维的方式。PCA算法相当于对数据做了一次整体的分析，找出最能代表这些数据的特征（这些特征不一定是原始选定的特征，而因子分子FA则是在原有的特征中找出特征投影交大的一部分特征），并用这些特征去表示原始数据。用投影的理论来说，所有的物体不再向原来表示特征的坐标轴去投影，而是改为向新的特征对应的坐标轴去投影。

## 2. SVD实现降维
经典的SVD公式：
$$
M = U\Sigma V^{T}
$$
假设$M\in R^{m\times n},则U\in R^{m\times k}, \Sigma\in R^{k\times k},V\in R^{n\times k}$(其中$k=min(m, n)，U、V$是正交单位矩阵, $\Sigma$是对角矩阵)。
正交单位矩阵等价于一组正交基，其作用是可以对object在对应的特征方向做投影。不考虑物理意义的情况下，在等式两边同时乘以$U^T$得$U^TM=\Sigma V^T$ （或者$MV=U\Sigma$），即实现了矩阵在不同表象下的表示。另外，$\Sigma$中的特征值的大小代表了其对应的特征在数据集中所占的比例大小，所以结合PCA的定义，只需从SVD分解后的结果中找出较大特征值对应的特征向量对原数据做投影即可实现降维的效果。
**至于选U中的特征向量还是V中的特征向量，取决于M中特征feature是在表示为行还是列，比如不同的行表示不同的样本、不同的列表示不同的特征时选用后一个。另外，不同的软件svd所得到的结果中V可能不太一致，有的已经做过转置，对比行列维度即可。**

## 3. opencv中sift特征降维与匹配
sift特征算是计算机视觉中较常用的一种特征表示，出于其旋转、缩放的不变性（这两个性质与其计算方式是相关的，缩放：通过前后层高斯金字塔计算，旋转：最后的128维特征都根据特定的规则选择主方向）在特征匹配中十分受欢迎。
python中sift特征计算：
```python
import cv2
img1 = cv2.imread('test1.jpg',0)# queryImage,参数0表示读取灰度图
img2 = cv2.imread("test2.jpg",0) # trainImage

# Initiate SIFT detector
sift = cv2.SIFT()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
#此处有两个返回值，
#其中kp包含sift特征的方向、位置、大小等信息
#其中des的shape为（sift_nb，128），sift_nb表示图像中检测到的sift特征数量
```
通过sift对两幅图片进行匹配：
```python
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
fine_match = []
for m,n in matches:
    if m.distance < 0.69*n.distance:
	    fine_match.append(m)
#对knn匹配的特征做判断，满足条件的定义为一个好的匹配点
#此处也可以借助RANSAC筛选
```
此处仅对两幅图片的特征做了比较，假设sift特征数分别为$sift\_nb\_1$,$ sift\_nb\_2$，则这一次匹配匹配的计算量$>sift\_nb\_1 * sift\_nb\_2 * 128 *n $, 对于size为（200x100）左右的图片，其特征数已经可以超过100。假如图片更大一下，带检索的目标图片数量增加时，计算量将变得更加庞大。故此处采用SVD的方式实现PCA降维sift特征，从而加快计算。
***首先需要解释的疑问是，sift特征本身是具有方向性的（旋转不变性的本质），降维以后是否还能保持方向性。PCA或者是SVD降维的本质是从高维向低维投影的算法，只需要对匹配的目标数据和待匹配数据做相同的投影，则得到的特征表示仍然保留了sift特征的方向性。***
使用SVD实现PCA降维sift特征：
```python
#des2_all = np.concatenate(des2_list,axis=0) #对多副图，先concat所有sift特征
#U,S,V = np.linalg.svd(des2_all)
U,S,V = np.linalg.svd(des2)
size_projected = 32
projector = V[:,:size_projected]
#for i in range(len(des2_list)):
#    des2_list[i] = des2_list[i].dot(projector)
des2_new = des2.dot(projector)
des1_new = des1.dot(projector)
```
对两组sift特征使用相同的特征向量投影，即获得了降维后的sift特征表示。实验中，第30维的特征值小于第一维的0.1倍（第30维的信息量已经少于第一维特征的10%），选取32维后时间花费上少了一倍多，而对结果的影响只有1~2%。PCA中的另外一种选择降维后的维度的方式是，看前多少维能代表一个合适比例的，假设为90%，维数k的选取规则为：
$$
min_{k} \frac{\Sigma _{i}^k\lambda_{i}}{\Sigma_{j}\lambda_{j}}>90\%
$$
>*第一次写博客，手贱出去点了一下保存的草稿查看markdown语法，直接把未保存的博客全覆盖了，一下信凉了半截。想到新点开编辑的博客时，貌似是把之前的网页直接关闭的。然后恢复关闭的网页，万幸。。。*
