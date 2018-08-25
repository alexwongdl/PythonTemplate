"""
Created by Alex Wang
On 2018-07-26
"""
import os
import sys

import numpy as np
import cv2

sys.path.append('../')

import face_alignment
from mtcnn import mtcnn_wrapper

root_dir = '/Users/alexwang/data/face_test'
img_dir = os.path.join(root_dir, 'aiguangjie')


# img_dir = os.path.join(root_dir, 'timeline')

def test_face_alignment():
    i = 1
    for img_name in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, img_name))
        face_img_list = mtcnn_wrapper.face_detect(img)
        for (face_img, new_rect, score, result) in face_img_list:
            keypoints = result['keypoints']
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            print(new_rect)
            print(result)
            aligned_face, M = face_alignment.align(img, new_rect, left_eye, right_eye)

            combined_img = np.concatenate((face_img, aligned_face), axis=1)
            i += 1

            cv2.imshow('result1:{}'.format(i), combined_img)
            # cv2.imshow('result2:{}'.format(i), aligned_face)

        key = cv2.waitKey(0) & 0xff
        if key == 27:
            sys.exit(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    test_face_alignment()
