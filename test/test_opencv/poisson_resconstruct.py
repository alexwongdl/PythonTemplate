"""
Created by Alex Wang on 2018-2-27
Poisson Rescontruct
https://gist.github.com/jackdoerner/b9b5e62a4c3893c76e4c

"""

import cv2

def mask_det(maskpath):      # maskpath为截取的水印样本
    img = cv2.imread(maskpath,0)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    sobelxy = numpy.sqrt(sobely**2+sobelx**2)
    mask = cv2.imread(maskpath)
    img_dst = poisson_reconstruct(sobely,sobelx,sobelxy)    #生成水印掩码
    # cv2.imwrite('crop/douyin2/mask_xy3.png',img_dst)     #保存的过程是对图像数据作了范围限制
    # mask1 = cv2.imread('crop/douyin2/mask_xy3.png',0)
    # for i in range(0,mask1.shape[0]):
    #     for j in range(0,mask1.shape[1]):
    #         if mask1[i,j]<80:#50    #对掩码设置阈值以精细化水印区域，阈值根据不同水印可微调
    #             mask[i,j,:]=[0,0,0]
    # cv2.imwrite('crop/douyin2/mask_new3.jpg',mask)   #保存水印模板
    return img_dst

"""
poisson_reconstruct.py
Fast Poisson Reconstruction in Python
Copyright (c) 2014 Jack Doerner
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import math
import numpy
import scipy, scipy.fftpack

def poisson_reconstruct(grady, gradx, boundarysrc):
    # Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
    # Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

    # Laplacian
    gyy = grady[1:,:-1] - grady[:-1,:-1]
    gxx = gradx[:-1,1:] - gradx[:-1,:-1]
    f = numpy.zeros(boundarysrc.shape)
    f[:-1,1:] += gxx
    f[1:,:-1] += gyy

    # Boundary image
    boundary = boundarysrc.copy()
    boundary[1:-1,1:-1] = 0;

    # Subtract boundary contribution
    f_bp = -4*boundary[1:-1,1:-1] + boundary[1:-1,2:] + boundary[1:-1,0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
    f = f[1:-1,1:-1] - f_bp

    # Discrete Sine Transform
    tt = scipy.fftpack.dst(f, norm='ortho')
    fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

    # Eigenvalues
    (x,y) = numpy.meshgrid(range(1,f.shape[1]+1), range(1,f.shape[0]+1), copy=True)
    denom = (2*numpy.cos(math.pi*x/(f.shape[1]+2))-2) + (2*numpy.cos(math.pi*y/(f.shape[0]+2)) - 2)

    f = fsin/denom

    # Inverse Discrete Sine Transform
    tt = scipy.fftpack.idst(f, norm='ortho')
    img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

    # New center + old boundary
    result = boundary
    result[1:-1,1:-1] = img_tt

    return result