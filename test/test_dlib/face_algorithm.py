"""
Created by Alex Wang
On 2018-05-03

CNN Face Detector
Face Alignment
Face Clustering
Face Detector
Face Jittering/Augmentation
Face Landmark Detection
Face Recognition
"""

import dlib
from skimage import io
import cv2


def test_cnn_face_detector():
    """
    You can get the mmod_human_face_detector.dat file from:\n
    http://dlib.net/files/mmod_human_face_detector.dat.bz2
    :return:
    """
    img_path = 'data/running_man.jpg'
    cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

    img = io.imread(img_path)
    # upsample image 1 time, make everything bigger and allow us to detect more faces.
    dets = cnn_face_detector(img, 2)

    img_cv2 = cv2.imread(img_path)
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
        cv2.rectangle(img_cv2, (d.rect.left(), d.rect.top()),
                      (d.rect.right(), d.rect.bottom()), color=(255, 0, 0), thickness=3)
    cv2.imshow('img', img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_hog_face_detector():
    """
    blog:
        https://blog.gtwang.org/programming/python-opencv-dlib-face-detection-implementation-tutorial/
    detector idx:
        https://blog.gtwang.org/programming/python-opencv-dlib-face-detection-implementation-tutorial/
    :return:
    """
    img_path = 'data/running_man.jpg'
    img = cv2.imread(img_path)
    hog_face_detector = dlib.get_frontal_face_detector()
    # 第二个参数--upsample, 第三个参数--threshold,超过threshold的结果会被输出
    # returns:rects--检测框位置,scores--得分,idx--检测器编号(正脸/侧脸等)
    rects, scores, idx = hog_face_detector.run(img, 2, 0)

    for i, rect in enumerate(rects):
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
        cv2.putText(img, '{:.2f}({})'.format(scores[i], idx[i]), (x1, y1),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # test_cnn_face_detector()
    test_hog_face_detector()