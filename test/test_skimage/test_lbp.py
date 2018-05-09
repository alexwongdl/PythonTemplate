"""
Created by Alex Wang on 2018-04

LBP常用使用方法
(1）首先将检测窗口划分为16×16的小区域（cell）；
(2)对于每个cell中的一个像素，将相邻的8个像素的灰度值与其进行比较，若周围像素值大于中心像素值，则该像素点的位置被标记为1，否则为0。这样，3*3邻域内的8个点经比较可产生8位二进制数，即得到该窗口中心像素点的LBP值；
(3)然后计算每个cell的直方图，即每个数字（假定是十进制数LBP值）出现的频率；然后对该直方图进行归一化处理。
(4)最后将得到的每个cell的统计直方图进行连接成为一个特征向量，也就是整幅图的LBP纹理特征向量；
(5)然后便可利用SVM或者其他机器学习算法进行分类了。

skimage.feature.local_binary_pattern(image, P, R, method='default')
P : int
Number of circularly symmetric neighbour set points (quantization of the angular space).

R : float
Radius of circle (spatial resolution of the operator).
method : {‘default’, ‘ror’, ‘uniform’, ‘var’}
default: gray scale, not rotation invariant
ror: gray scale and rotation invariant
uniform: uniform patters
nri_uniform: non rotation-invariant uniform patterns
"""

import cv2
from skimage.feature import local_binary_pattern

radius = 3
n_points = radius * 8

google_img = cv2.imread('google.jpg')
google_gray = cv2.cvtColor(google_img, cv2.COLOR_BGR2GRAY)

lbp_ror = local_binary_pattern(google_gray, n_points, radius, 'ror')
lbp_uniform = local_binary_pattern(google_gray, n_points, radius, 'uniform')

cv2.imshow('org_image', google_img)
cv2.imshow('lbp_ror', lbp_ror)
cv2.imshow('lbp_uniform', lbp_uniform)
cv2.waitKey(0)
cv2.destroyAllWindows()



