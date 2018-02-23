"""
Created by Alex Wang on 2018-02-23
"""
import cv2

def test_harris():
    """
    Harris角点检测
    :return:
    """
    img = cv2.imread('dl.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    harris_corners = cv2.cornerHarris(img_gray, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    img[harris_corners > harris_corners.max() * 0.01] = [0,0,225]
    print(harris_corners.max())
    print(harris_corners)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_harris()