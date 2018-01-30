"""
Created by Alex Wang
On 2018-01-30
"""

import cv2

def test_imshow():
    """
    测试opencv图像展示
    :return:
    """
    img = cv2.imread('E://blog/oxford_pet.png')
    cv2.imshow('window1', img)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_imshow()