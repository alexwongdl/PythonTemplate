"""
Created by Alex Wang on 2018-02-03
"""

import cv2
import numpy as np

from matplotlib import pyplot as plt

def color_space():
    """
    打印所有可能的颜色空间转换
    :return:
    """
    flags = [i for i in dir(cv2) if i.startswith('COLOR_BGR')]
    [print(flag) for flag in flags]

def image_resize():
    """
    图像缩放
    :return:
    """
    img = cv2.imread('dl.jpg')

    # img_resize_factor = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_AREA)
    img_resize_factor = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_LINEAR)
    cv2.imshow('dl_refactor',img_resize_factor)

    rows, cols = img.shape[:2]
    img_resize = cv2.resize(img, (int(0.5 * cols), int(0.5 * rows)), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('dl_resize', img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_transformation():
    """
    图像平移、旋转、仿射变换
    :return:
    """
    img = cv2.imread('dl.jpg')
    rows, cols = img.shape[:2]
    # 平移
    shift_mat = np.array([[1,0,20],[0,1,50]], dtype=np.float32) # x方向平移20像素，y方向移动50像素
    img_shift = cv2.warpAffine(img, shift_mat, (cols + 20, rows + 50))  #最后一个参数是输出图像大小
    cv2.imshow('img_shift', img_shift)

    # 旋转
    rotate_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    # rotate_mat = cv2.getRotationMatrix2D((0, 0), 10, 1)
    print(rotate_mat)
    img_rotate =  cv2.warpAffine(img, rotate_mat, (cols + 30, rows + 30))
    cv2.imshow('img_rotate', img_rotate)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_kernel_conv():
    """
    图像卷积：模糊、
    :return:
    """
    img = cv2.imread('dl.jpg')
    avg_kernel = np.ones((5,5), dtype=np.float32) / 25
    # 应用kernel
    img_avg = cv2.filter2D(img, -1, avg_kernel)
    # 均值滤波
    img_blur = cv2.blur(img, (5,5))
    # 高斯滤波
    gaussian_kernel = cv2.getGaussianKernel(3, 1)
    print('gaussian_kernel:', gaussian_kernel)
    gaussian_img = cv2.GaussianBlur(img, (3,3), sigmaX=1, sigmaY = 2)
    # 中值滤波
    median_img = cv2.medianBlur(img, 5)
    #双边滤波
    bilateral_img = cv2.bilateralFilter(img, 9, 75, 75)

    cv2.imshow('org_img', img)
    cv2.imshow('img_avg', img_avg)
    cv2.imshow('img_blur', img_blur)
    cv2.imshow('img_gaussian', gaussian_img)
    cv2.imshow('img_median', median_img)
    cv2.imshow('img_bilateral', bilateral_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_edge_detect():
    """
    图像边缘检测：Sobel、Laplacian、Canny
    :return:
    """
    img = cv2.imread('dl.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel
    gaussian_img = cv2.GaussianBlur(img_gray, (3,3), sigmaX=1, sigmaY = 1)
    img_sobelx = cv2.Sobel(gaussian_img, cv2.CV_8U, dx = 1, dy = 0, ksize = 3)
    img_sobely = cv2.Sobel(gaussian_img, cv2.CV_8U, dx = 0, dy = 1, ksize = 3)
    img_sobel_weight = cv2.addWeighted(img_sobelx, 0.5, img_sobely, 0.5, 0)
    img_sobelxy = cv2.Sobel(gaussian_img, cv2.CV_8U, dx = 1, dy = 1, ksize = 3)

    # Laplacian
    img_laplacian = cv2.Laplacian(gaussian_img, cv2.CV_64F)

    # Canny
    img_canny = cv2.Canny(img, 50, 150, L2gradient = True)

    cv2.imshow('img_gray', img_gray)
    cv2.imshow('img_sobelx', img_sobelx)
    cv2.imshow('img_sobely', img_sobely)
    cv2.imshow('img_sobelxy', img_sobel_weight)
    cv2.imshow('img_laplacian', img_laplacian)
    cv2.imshow('img_canny', img_canny)
    # cv2.imshow('img_sobelxy_1', img_sobelxy)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_histogram():
    """
    测试颜色直方图
    :return:
    """
    img = cv2.imread('dl.jpg')
    colors = ['b','g','r']
    for i, color in enumerate(colors):
        plt.plot(cv2.calcHist([img], [i], None, [256], [0, 256]), color)
    plt.show()

def hough_line_detect():
    """
    hough直线检测
    :return:
    """
    # cv2.HoughLines
    img = cv2.imread('dl.jpg')
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_canny  = cv2.Canny(img_gray, 50, 150, L2gradient = True)
    lines = cv2.HoughLines(img_canny, 1, np.pi/180, 200)
    print(lines[:,0,:])

    for rho, theta in lines[:,0,:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)

    # cv2.HoughLinesP
    lines_p = cv2.HoughLinesP(img_canny, 1, np.pi/180, 100, minLineLength=20, maxLineGap=1)
    for x1, y1, x2, y2 in lines_p[:,0,:]:
        cv2.line(img_copy, (x1, y1), (x2, y2), (0,0,255), 2)

    cv2.imshow('img_canny', img_canny)
    cv2.imshow('hough_lines', img)
    cv2.imshow('hough_lines_p', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == '__main__':
    color_space()
    # image_resize()
    # test_transformation()
    # test_kernel_conv()
    # test_edge_detect()
    # test_histogram()
    hough_line_detect()