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

def test_sift():
    """
    SIFT 特征提取
    :return:
    """
    img = cv2.imread('dl.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img_gray, None)
    img_kp = img.copy()
    cv2.drawKeypoints(img, kp, img_kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 获取SIFT描述子
    kp, sift_des = sift.compute(img_gray, kp)
    print(sift_des.shape, sift_des)

    kp_1, sift_des_1 = sift.detectAndCompute(img_gray, None)
    print(sift_des_1.shape, sift_des_1)

    cv2.imshow('img_kp', img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # test_harris()
    test_sift()