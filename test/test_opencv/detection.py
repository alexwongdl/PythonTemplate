"""
Created by Alex Wang on 2018-03-15
"""
import cv2
import os

print(os.getcwd())  ##当前目录


def opencv_face_detect():
    """
    opencv haar cascade face detection
    :return:
    """
    i = 0
    img_path = os.path.join(os.getcwd(), 'data', 'running_man.jpg')
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    print(dir(face_cascade))
    print(help(face_cascade.detectMultiScale))
    faces = face_cascade.detectMultiScale(img_gray, 1.1, 3)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)
        roi_gray = img_gray[y: y + h, x:x + w]
        roi_img = img[ y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_img, (ex, ey), (ex + ew, ey + eh), color=(0, 0, 255), thickness=3)
        # cv2.imshow('img_' + str(i), roi_img.copy())
        # i += 1

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def opencv_pedestrian_detect():
    """
    opencv hog + svm pedestrian detection
       实际上，在运用的时候，我们通常是选取一幅图像中的一个窗口来进行特征提取，依然以上述220X310大小图像为例，
       经过缩放处理后为216x304，但并不直接提取整个图像的HOG特征，而是用一个固定大小的窗口在图像上滑动，
       滑动的间隔为8个像素，opencv中默认的窗口大小为128x64（高128，宽64），即有(128÷8)x(64÷8)=16x8个cell，
       也即有15x7个block，这样一来一幅图像就可以取到(27-16)x(38-8)=11x30=330个窗口。现在提取每个窗口的HOG特征，
       则可得到105x36=3780维HOG特征向量。https://blog.csdn.net/hujingshuang/article/details/47337707/

       references:https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
    :return:
    """
    img = cv2.imread('data/pedestrian.jpg')
    print('shape of img',img.shape)
    # hog feature extraction  'HOGDescriptor', 'HOGDescriptor_DEFAULT_NLEVELS', 'HOGDescriptor_L2Hys', 'HOGDescriptor_getDaimlerPeopleDetector', 'HOGDescriptor_getDefaultPeopleDetector'
    print(help(cv2.HOGDescriptor))
    win_size = (64, 128)
    block_size = (16,16)
    block_stride = (8,8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    locations, weights = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for location, weight in zip(locations, weights):
        (x, y, w, h) = location
        cv2.rectangle(img, (x, y), (x+ w, y + h), (255,0,0), 2)
        print(weight)
    ## non_max_suppression : https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
    print(help(hog.detectMultiScale))

    cv2.imshow('pedestrian', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # opencv_face_detect()
    opencv_pedestrian_detect()
