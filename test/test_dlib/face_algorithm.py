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

def test_face_detector():
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

if __name__ == '__main__':
    test_face_detector()