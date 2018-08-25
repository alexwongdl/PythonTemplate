"""
Created by Alex Wang
On 2018-07-26
reference:https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
            imutls:https://github.com/jrosebr1/imutils

Many facial recognition algorithms, including Eigenfaces, LBPs for face recognition, Fisherfaces, and deep learning/metric methods can all benefit from applying facial alignment before trying to identify the face
"""
import numpy as np
import cv2


def align(img_org, rect, left_eye, right_eye):
    """
    :param img: opencv BGR format image
    :param rect: rect = (x_min, y_min, x_max, y_max)
    :param left_eye: (cols, rows)
    :param right_eye: (cols, rows)
    :return:
    """
    img = img_org.copy()
    height, width = img.shape[0:2]
    (x_min, y_min, x_max, y_max) = rect

    # compute the angle between the eye centroids
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    # print('angle:{}'.format(angle))

    eyesCenter = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)  # (cols, rows)


    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale=1)
    rotate_img = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC)

    aligned_face = rotate_img[y_min:y_max, x_min: x_max, :]

    # plot to debug
    # cv2.circle(img, center=(left_eye[0], left_eye[1]), radius=5, color=(0, 255, 0), thickness=2)
    # cv2.circle(img, center=(right_eye[0], right_eye[1]), radius=5, color=(0, 0, 255), thickness=2)
    # cv2.circle(img, center=(eyesCenter[0], eyesCenter[1]), radius=5, color=(255, 0, 0), thickness=2)
    # cv2.imshow('img_org', img)
    # cv2.imshow('rotate_img', rotate_img)

    # left_eye = apply_transform(left_eye, M)
    # right_eye = apply_transform(right_eye, M)
    # eyesCenter = apply_transform(eyesCenter, M)
    # cv2.circle(rotate_img, center=(left_eye[0], left_eye[1]), radius=5, color=(0, 255, 0), thickness=2)
    # cv2.circle(rotate_img, center=(right_eye[0], right_eye[1]), radius=5, color=(0, 0, 255), thickness=2)
    # cv2.circle(rotate_img, center=(eyesCenter[0], eyesCenter[1]), radius=5, color=(255, 0, 0), thickness=2)
    # cv2.imshow('rotate_img', rotate_img)

    return aligned_face, M

def apply_transform(point, M):
    """
    :param point:(x,y) point
    :param M:
    :return:
    """
    rotated_point = M.dot(np.array(point + (1,))).astype(np.int32)
    return rotated_point