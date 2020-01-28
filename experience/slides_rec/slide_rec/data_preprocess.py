"""
Created by Alex Wang on 2019-03-17
"""
import math
import time
import numpy as np
from slide_rec_config import CROP_SIZE, CLIP_LENGTH
import cv2

SHOT_NUM = CLIP_LENGTH


def shot_data_prepare(shot_data_batch):
    """
    :param shot_data_batch:
    :return: frames_batch -- [batch_size, shot_num, 224, 224, 3]
             labels -- [batch_size]
    """

    labels = np.zeros(shape=(len(shot_data_batch)), dtype=np.int64)
    video_ids = np.zeros(shape=(len(shot_data_batch)), dtype=np.int64)
    frames_batch = []
    shot_idx = 0
    for shot_data in shot_data_batch:
        frame_matrix = np.zeros(shape=(SHOT_NUM, CROP_SIZE, CROP_SIZE, 1), dtype=np.float32)
        label = 0
        video_id = 0
        frame_idx = 0

        for shot in shot_data:  # k frame
            frame, label, video_id = shot['frame'], shot['label'], shot['video_id']
            frame_matrix[frame_idx, :, :, :] = frame
            frame_idx += 1

        labels[shot_idx] = label
        video_ids[shot_idx] = video_id
        frames_batch.append(frame_matrix)
        shot_idx += 1

    return np.asarray(frames_batch), labels, video_ids


def shot_data_prepare_gray(shot_data_batch):
    """
    return gray images
    :param shot_data_batch:
    :return: frames_batch -- [batch_size, shot_num, 224, 224, 1]
             labels -- [batch_size]
    """

    labels = np.zeros(shape=(len(shot_data_batch)), dtype=np.int64)
    video_ids = np.zeros(shape=(len(shot_data_batch)), dtype=np.int64)
    frames_batch = []
    shot_idx = 0
    for shot_data in shot_data_batch:
        frame_matrix = np.zeros(shape=(SHOT_NUM, CROP_SIZE, CROP_SIZE, 1), dtype=np.float32)
        label = 0
        video_id = 0
        frame_idx = 0

        for shot in shot_data:  # k frame
            frame, label, video_id = shot['frame'], shot['label'], shot['video_id']
            frame_matrix[frame_idx, :, :, :] = np.expand_dims(frame, -1)
            frame_idx += 1

        labels[shot_idx] = label
        video_ids[shot_idx] = video_id
        frames_batch.append(frame_matrix)
        shot_idx += 1

    return np.asarray(frames_batch), labels, video_ids


def one_shot_optical(shot_data):
    frame_matrix = np.zeros(shape=(SHOT_NUM, CROP_SIZE, CROP_SIZE, 1), dtype=np.float32)
    optical_matrix = np.zeros(shape=(SHOT_NUM, CROP_SIZE, CROP_SIZE, 2), dtype=np.float32)
    label = 0
    video_id = 0
    frame_idx = 0

    for shot in shot_data:  # k frame
        frame, label, video_id = shot['frame'], shot['label'], shot['video_id']
        frame_matrix[frame_idx, :, :, :] = np.expand_dims(frame, -1)
        frame_idx += 1

    for i in range(1, frame_idx):
        prev = frame_matrix[i - 1, :, :, :]
        next = frame_matrix[i, :, :, :]
        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # hsv = np.zeros_like(prev)
        # hsv[..., 1] = 255
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = ang * 180 / np.pi / 2
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # image = (rgb / 255. - 0.5) * 2
        optical_matrix[i - 1, :, :, :] = flow

    return (label, video_id, optical_matrix)


def shot_data_prepare_optical(shot_data_batch):
    """
    return optical flow images
    :param shot_data_batch:
    :return: frames_batch -- [batch_size, shot_num, 224, 224, 1] gray image
             labels -- [batch_size]
    """

    labels = np.zeros(shape=(len(shot_data_batch)), dtype=np.int64)
    video_ids = np.zeros(shape=(len(shot_data_batch)), dtype=np.int64)
    frames_batch = []
    optical_batch = []
    shot_idx = 0

    for shot_data in shot_data_batch:
        frame_matrix = np.zeros(shape=(SHOT_NUM, CROP_SIZE, CROP_SIZE, 1), dtype=np.float32)
        optical_matrix = np.zeros(shape=(SHOT_NUM, CROP_SIZE, CROP_SIZE, 2), dtype=np.float32)
        label = 0
        video_id = 0
        frame_idx = 0  # frame_num

        for shot in shot_data:  # k frame
            frame, label, video_id = shot['frame'], shot['label'], shot['video_id']
            frame_matrix[frame_idx, :, :, :] = np.expand_dims(frame, -1)
            frame_idx += 1

        for i in range(1, frame_idx):
            prev = frame_matrix[i - 1, :, :, :]
            next = frame_matrix[i, :, :, :]
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # hsv = np.zeros_like(prev)
            # hsv[..., 1] = 255
            # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # hsv[..., 0] = ang * 180 / np.pi / 2
            # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            # image = (rgb / 255. - 0.5) * 2
            optical_matrix[i - 1, :, :, :] = flow

        labels[shot_idx] = label
        video_ids[shot_idx] = video_id
        # frames_batch.append(frame_matrix)
        optical_batch.append(optical_matrix)
        shot_idx += 1

    return np.asarray(optical_batch), labels, video_ids


def shot_data_prepare_optical_max_5(shot_data_batch, return_frame = False, pre_extract = False):
    """
    return optical flow images, maximum use 5 frames
    :param shot_data_batch:
    :return: optical_matrix -- [batch_size, 4, 224, 224, 1] gray image
             labels -- [batch_size]
    """
    labels = np.zeros(shape=(len(shot_data_batch)), dtype=np.int64)
    video_ids = np.zeros(shape=(len(shot_data_batch)), dtype=np.int64)
    shot_ids = np.zeros(shape=(len(shot_data_batch)), dtype=np.int64)

    optical_batch = []
    frames_batch = []
    shot_idx = 0
    total_cost_time = 0.

    for shot_data in shot_data_batch:
        time_3 = time.time()
        frame_matrix = np.zeros(shape=(5, CROP_SIZE, CROP_SIZE, 1), dtype=np.float32)
        optical_matrix = np.zeros(shape=(4, CROP_SIZE, CROP_SIZE, 2), dtype=np.float32)
        label = -1
        video_id = -1
        shot_id = -1
        frame_num = len(shot_data)
        frame_idx = 0
        if frame_num > 20:
            print('error: shot_data_prepare_optical_max_5 frame_num:{} > 20'.format(frame_num))
        time_3_1 = time.time()

        selected_frame_index = frame_select(frame_num, 5)
        time_3_2 = time.time()
        for idx in selected_frame_index:
            shot = shot_data[idx]
            frame, label, video_id, shot_id = shot['frame'], shot['label'], shot['video_id'], shot['shot_id']
            frame_matrix[frame_idx, :, :, :] = np.expand_dims(frame, -1)
            frame_idx += 1
        time_4 = time.time()

        time_1 = time.time()
        for i in range(1, frame_idx):
            prev = frame_matrix[i - 1, :, :, :]
            next = frame_matrix[i, :, :, :]
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # hsv = np.zeros_like(prev)
            # hsv[..., 1] = 255
            # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # hsv[..., 0] = ang * 180 / np.pi / 2
            # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            # image = (rgb / 255. - 0.5) * 2
            optical_matrix[i - 1, :, :, :] = flow

        time_2 = time.time()
        total_cost_time += time_2 - time_1
        labels[shot_idx] = label
        video_ids[shot_idx] = video_id
        shot_ids[shot_idx] = shot_id
        if return_frame:
            frames_batch.append(frame_matrix)
        optical_batch.append(optical_matrix)
        shot_idx += 1

    if return_frame:
        return np.asarray(optical_batch), np.asarray(frames_batch), labels, video_ids
    if pre_extract:
        return np.asarray(optical_batch), labels, video_ids, shot_ids

    return np.asarray(optical_batch), labels, video_ids


def shot_data_prepare_optical_8(shot_data_batch):
    """
    return optical flow images, maximum use 5 frames, generate 8 shot feature
    :param shot_data_batch:
    :return: optical_matrix -- [batch_size, 4, 224, 224, 1] gray image
             labels -- [batch_size]
    """
    labels = []
    video_ids = []
    optical_batch = []

    compare_tuples = [(0, 2), (1, 3), (2, 4), (0, 4)]

    for shot_data in shot_data_batch:
        frame_matrix = np.zeros(shape=(5, CROP_SIZE, CROP_SIZE, 1), dtype=np.float32)
        optical_matrix = np.zeros(shape=(8, CROP_SIZE, CROP_SIZE, 2), dtype=np.float32)
        label = 0
        video_id = 0
        frame_num = len(shot_data)
        if frame_num < 5:
            continue
        frame_idx = 0

        selected_frame_index = frame_select(frame_num, 5)
        for idx in selected_frame_index:
            shot = shot_data[idx]
            frame, label, video_id = shot['frame'], shot['label'], shot['video_id']
            frame_matrix[frame_idx, :, :, :] = np.expand_dims(frame, -1)
            frame_idx += 1

        for i in range(1, frame_idx):
            prev = frame_matrix[i - 1, :, :, :]
            next = frame_matrix[i, :, :, :]
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            optical_matrix[i - 1, :, :, :] = flow

        optical_idx = 4
        for prev_idx, next_idx in compare_tuples:
            prev = frame_matrix[prev_idx, :, :, :]
            next = frame_matrix[next_idx, :, :, :]
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            optical_matrix[optical_idx, :, :, :] = flow
            optical_idx += 1

        labels.append(label)
        video_ids.append(video_id)
        # frames_batch.append(frame_matrix)
        optical_batch.append(optical_matrix)

    return np.asarray(optical_batch), np.asarray(labels, dtype=np.int64), \
           np.asarray(video_ids, dtype=np.int64)


def frame_select(frame_num, select_num):
    arr = range(frame_num)
    step = frame_num * 1.0 / select_num
    new_arr = []
    arr_idx = 0
    selected_idx_set = set()
    for i in range(select_num):
        idx = int(math.ceil(arr_idx))
        if idx not in selected_idx_set and idx < frame_num:
            new_arr.append(arr[idx])
            selected_idx_set.add(idx)
        arr_idx += step
    return new_arr


def test_frame_select():
    target_num = 10

    for frame_num in range(1, 21):
        new_arr = frame_select(frame_num, target_num)
        print('frame num:{}, arr:{}'.format(frame_num, new_arr))


if __name__ == '__main__':
    test_frame_select()
