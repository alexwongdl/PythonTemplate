"""
Created by Alex Wang
On 2018-07-25

threshold:0.98 
"""
import traceback

import cv2
from mtcnn import MTCNN

mtcnn_detector = MTCNN(weights_file='/Users/alexwang/workspace/video/face/mtcnn/data/mtcnn_weights.npy')


def face_detect(img, expand=True, max_boundary=480):
    """
    过滤掉宽高比过大的图片:1:3/3:1
    :param img: rgb format image
    :param max_boundary: if None , not resize
    :return:(face, new_rect, score, result) pair list, new_rect = (x_min, y_min, x_max, y_max)

    [{'box': [277, 90, 48, 63], 'confidence': 0.9985162615776062, 'keypoints': {'left_eye': (291, 117), 'mouth_left': (296, 143), 'nose': (303, 131), 'mouth_right': (313, 141), 'right_eye': (314, 114)}}]
    """
    height, width = img.shape[0:2]
    ratio = 1.
    new_height, new_width = img.shape[0:2]
    img_resize = img.copy()
    if max_boundary:
        ratio = max_boundary * 1.0 / max(height, width)
        new_height = int(height * ratio)
        new_width = int(width * ratio)
        img_resize = cv2.resize(img_resize, (new_width, new_height))

    results = mtcnn_detector.detect_faces(img_resize)
    face_img_list = []
    for result in results:
        try:
            rect = result['box']
            score = result['confidence']
            x, y = rect[0], rect[1]
            w = rect[2]
            h = rect[3]
            wh_ratio = w * 1.0 / h
            if wh_ratio > 3 or wh_ratio < 0.33:
                print('filt face wh_ratio > 3 or wh_ratio < 0.33')
                continue
            up_margin = 0
            margin = 0
            if expand:
                up_margin = w / 2.5
                margin = w / 4
            y_min = int(max(0, y - up_margin) / ratio)
            y_max = int(min(height, y + h + margin) / ratio)
            x_min = int(max(0, x - margin) / ratio)
            x_max = int(min(width, x + w + margin) / ratio)
            new_rect = (x_min, y_min, x_max, y_max)

            # change result with ratio
            result['box'] = [int(x / ratio), int(y / ratio), int(w / ratio), int(h / ratio)]
            key_points = result['keypoints']
            new_keypoints = {}
            for key, value in key_points.items():
                value_x = int(value[0] / ratio)
                value_y = int(value[1] / ratio)
                new_keypoints[key] = (value_x, value_y)
            result['keypoints'] = new_keypoints

            # debug
            # keypoints = result['keypoints']
            # left_eye = keypoints['left_eye']
            # right_eye = keypoints['right_eye']
            # nose = keypoints['nose']
            # eyesCenter = ((left_eye[0] + right_eye[0]) // 2,
            #               (left_eye[1] + right_eye[1]) // 2)  # (cols, rows)
            # cv2.circle(img, center=(left_eye[0], left_eye[1]), radius=5, color=(0, 255, 0), thickness=2)
            # cv2.circle(img, center=(right_eye[0], right_eye[1]), radius=5, color=(0, 0, 255), thickness=2)
            # cv2.circle(img, center=(eyesCenter[0], eyesCenter[1]), radius=5, color=(255, 0, 0), thickness=2)
            # cv2.circle(img, center=(nose[0], nose[1]), radius=5, color=(255, 0, 0), thickness=2)
            # debug end

            face_img = img[y_min:y_max, x_min: x_max, :].copy()
            face_img_list.append((face_img, new_rect, score, result))
        except Exception as e:
            traceback.print_exc()
    return face_img_list
