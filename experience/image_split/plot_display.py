"""
Created by Alex Wang 2018-04-26
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def convert_img(img):
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r,g,b])
    return img2

def plot_image_split_result_3():
    print('current_dir:', os.getcwd())
    data_dir = '/Users/alexwang/workspace/video/image_split/split_result'
    print(os.path.isdir(data_dir))

    img_one_path = os.path.join(data_dir, 'img_org_resize.jpg')
    print(os.path.isfile(img_one_path))
    img_two_path = os.path.join(data_dir, 'img_patches2.jpg')
    img_three_path = os.path.join(data_dir, 'img_patches3.jpg')
    img_four_path = os.path.join(data_dir, 'img_patches4.jpg')
    img_canny_path = os.path.join(data_dir,'img_canny_plot.jpg')

    img_one = convert_img(cv2.imread(img_one_path))
    img_canny = convert_img(cv2.imread(img_canny_path))
    img_two = convert_img(cv2.imread(img_two_path))
    img_three = convert_img(cv2.imread(img_three_path))
    img_four = convert_img(cv2.imread(img_four_path))


    plt.subplot(333)
    plt.imshow(img_two)
    plt.axis('off')
    plt.title('sub_image_1')
    plt.subplot(336)
    plt.imshow(img_three)
    plt.axis('off')
    plt.title('sub_image_2')
    plt.subplot(339)
    plt.imshow(img_four)
    plt.axis('off')
    plt.title('sub_image_3')
    plt.subplot(131)
    plt.imshow(img_one)
    plt.axis('off')
    plt.title('origin_image')
    plt.subplot(132)
    plt.imshow(img_canny)
    plt.axis('off')
    plt.title('canny_image')
    plt.show()

def plot_image_split_result():

    data_dir = '/Users/alexwang/data/image_split/result_view_v2'

    # img_one_path = os.path.join(data_dir, 'TB2.2MCclsmBKNjSZFsXXaXSVXa_!!3432389200.jpg')
    # img_two_path = os.path.join(data_dir, 'TB2.2MCclsmBKNjSZFsXXaXSVXa_!!3432389200_72.jpg')
    # img_three_path = os.path.join(data_dir, 'TB2.2MCclsmBKNjSZFsXXaXSVXa_!!3432389200_73.jpg')

    # img_one_path = os.path.join(data_dir, 'TB21Vi2mL5TBuNjSspmXXaDRVXa_!!99203978.jpg')
    # img_two_path = os.path.join(data_dir, 'TB21Vi2mL5TBuNjSspmXXaDRVXa_!!99203978_86.jpg')
    # img_three_path = os.path.join(data_dir, 'TB21Vi2mL5TBuNjSspmXXaDRVXa_!!99203978_87.jpg')

    img_one_path = os.path.join(data_dir, 'TB1YfZhmH5YBuNjSspo8FYeNFXa')
    img_two_path = os.path.join(data_dir, 'TB1YfZhmH5YBuNjSspo8FYeNFX_70a')
    img_three_path = os.path.join(data_dir, 'TB1YfZhmH5YBuNjSspo8FYeNFX_71a')

    img_one = convert_img(cv2.imread(img_one_path))
    img_two = convert_img(cv2.imread(img_two_path))
    img_three = convert_img(cv2.imread(img_three_path))


    plt.subplot(222)
    plt.imshow(img_two)
    plt.axis('off')
    plt.title('sub_image_1')
    plt.subplot(224)
    plt.imshow(img_three)
    plt.axis('off')
    plt.title('sub_image_2')
    plt.subplot(121)
    plt.imshow(img_one)
    plt.axis('off')
    plt.title('origin_image')
    plt.show()


def example():
    import matplotlib.pyplot as plt
    import numpy as np


    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    # plt.axis([0, 6, 0, 20])
    # plt.show()

    # t = np.arange(0., 5., 0.2)
    # plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')


    def f(t):
        return np.exp(-t) * np.cos(2 * np.pi * t)


    t1 = np.arange(0, 5, 0.1)
    t2 = np.arange(0, 5, 0.02)

    plt.figure(12)
    plt.subplot(221)
    plt.plot(t1, f(t1), 'bo', t2, f(t2), 'r--')

    plt.subplot(222)
    plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')

    plt.subplot(212)
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

    plt.show()

if __name__ == '__main__':
    # plot_image_split_result()
    plot_image_split_result_3()
    # example()