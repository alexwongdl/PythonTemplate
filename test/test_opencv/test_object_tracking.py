"""
Created by Alex Wang
On 2018-07-20
"""
import os
import sys
import shutil
import traceback
import numpy as np

import cv2

from face import face_detect
import sort
import opencv_trackers


def rectangle(image, x, y, w, h, color, thickness=2, label=None):
    """Draw a rectangle.

    Parameters
    ----------
    x : float | int
        Top left corner of the rectangle (x-axis).
    y : float | int
        Top let corner of the rectangle (y-axis).
    w : float | int
        Width of the rectangle.
    h : float | int
        Height of the rectangle.
    label : Optional[str]
        A text label that is placed at the top left corner of the
        rectangle.
    """
    pt1 = int(x), int(y)
    pt2 = int(x + w), int(y + h)
    cv2.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), thickness)


def recreate_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def face_detect_frame():
    method_type = 'hog'
    video_root = '/Users/alexwang/data'
    video_list = ['14456458.mp4', '32974696.mp4', '16815015.mp4', '10616634.mp4']
    colours = np.random.rand(32, 3) * 256  # used only for display

    print('start process videos...')
    for video_name in video_list:
        video_tracker = sort.Sort(max_age=5)
        kal = sort.KalmanBoxTracker([0, 0, 1, 1, 0])
        kal.clear_count()
        cap = cv2.VideoCapture(os.path.join(video_root, video_name))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        output_path = os.path.join(video_root, 'notexpand_{}_{}'.format(method_type, video_name))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 15, (frame_width, frame_height))

        while (cap.isOpened()):
            try:
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if method_type == 'cnn':
                    face_img_list = face_detect.cnn_face_detect(image_rgb, expand=False)
                else:
                    face_img_list = face_detect.hog_face_detect(image_rgb, expand=False)
                detections = []
                for (face, rect, score) in face_img_list:
                    if score < 0.4:
                        continue
                    x_min, y_min, x_max, y_max = rect
                    detections.append([x_min, y_min, x_max, y_max, 10 * score])

                print('detections:', detections)
                track_bbs_ids = video_tracker.update(np.asarray(detections))
                for d in track_bbs_ids:
                    print('d:', d)

                    d = d.astype(np.int32)
                    rectangle(frame, d[0], d[1], d[2] - d[0],
                              d[3] - d[1], colours[d[4] % 32, :],
                              thickness=2, label=str(d[4]))

                cv2.imshow('image', frame)
                out.write(frame)
                # Exit if ESC oressed
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    sys.exit(0)
                elif key == ord('q'):
                    break
            except Exception as e:
                traceback.print_exc()
        cap.release()
        out.release()


def test_Kalman():
    kalman = sort.KalmanBoxTracker([260, 69, 78, 85, 1.2872009431190912])
    i = 0
    dets = np.asarray([[260, 69, 78, 85, 1.1129303132362787]])
    print(dets[i, :])


def test_opencv_tracker():
    method_type = 'hog'
    video_root = '/Users/alexwang/data'
    video_list = ['32974696.mp4', '14456458.mp4', '16815015.mp4', '10616634.mp4']
    colours = np.random.rand(32, 3) * 256  # used only for display

    print('start process videos...')
    for video_name in video_list:
        video_tracker = opencv_trackers.Trackers()
        cap = cv2.VideoCapture(os.path.join(video_root, video_name))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        output_path = os.path.join(video_root, 'notexpand_opencv_{}_{}'.format(method_type, video_name))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 15, (frame_width, frame_height))

        frame_idx = 0
        frame_gap = 3
        while (cap.isOpened()):
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame_idx % frame_gap == 1:
                    if method_type == 'cnn':
                        face_img_list = face_detect.cnn_face_detect(image_rgb, expand=False)
                    else:
                        face_img_list = face_detect.hog_face_detect(image_rgb, expand=False)
                detections = []
                for (face, rect, score) in face_img_list:
                    if score < 0.4:
                        continue
                    x_min, y_min, x_max, y_max = rect
                    detections.append([x_min, y_min, x_max, y_max, 10 * score])

                # print('detections:', detections)

                if frame_idx % frame_gap == 1:
                    track_bbs_ids = video_tracker.update_and_detect(frame, np.asarray(detections))
                else:
                    track_bbs_ids = video_tracker.update(frame)

                for tracker_info in track_bbs_ids:
                    d = tracker_info['box']
                    d = np.asarray(list(d))
                    print('d:', d)

                    d = d.astype(np.int32)
                    rectangle(frame, d[0], d[1], d[2] - d[0],
                              d[3] - d[1], colours[tracker_info['count'] % 32, :],
                              thickness=2, label=str(tracker_info['count']))

                cv2.imshow('image', frame)
                out.write(frame)
                # Exit if ESC oressed
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    sys.exit(0)
                elif key == ord('q'):
                    break
            except Exception as e:
                traceback.print_exc()
        cap.release()
        out.release()


if __name__ == "__main__":
    face_detect_frame()
    # test_Kalman()
    # test_opencv_tracker()
