"""
Created by Alex Wang on 2017-11-16
"""
import os
import cv2

def image_resolution(image):
    """
    图片清晰度，返回double值
    :param image:image = cv2.imread(image_path)获取的图片对象
    :return:
    """
    shape = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    fm_fomat = round(float(fm), 2)
    return fm_fomat


def cal_video_fm(video_feature):
    """
    计算一个视频最大的fm值
    :param video_feature:
    :return:
    """
    max_keyframe_check_num = 3
    keyfram_check_num = 0
    image_resolution_list = []
    print('keyframe_path: ' + video_feature['keyframe_path'])
    if os.path.isdir(video_feature['keyframe_path']):
        for file in os.listdir(video_feature['keyframe_path']):
            img_path = os.path.join(video_feature['keyframe_path'], file)
            if os.path.isfile(img_path) and keyfram_check_num < max_keyframe_check_num:
                keyfram_check_num += 1
                image = cv2.imread(img_path)
                image_resolution_list.append(image_resolution(image))
    if len(image_resolution_list )> 0:
        video_feature.update({'max_resolution_value': max(image_resolution_list)})
        video_feature.update({'mean_resolution_value': sum(image_resolution_list) * 1.0 / len(image_resolution_list)})
        video_feature.update({['norm_resolution_value']:video_feature['mean_resolution_value'] * (640.0 * 368) / (video_feature['video_height'] * video_feature['video_width']) })
    else:
        video_feature.update({'max_resolution_value': 500})
        video_feature.update({'mean_resolution_value': 500})
        print('image_resolution_list = 0, docid:{}'.format(video_feature['doc_id']))
        video_feature.update({['norm_resolution_value']:video_feature['mean_resolution_value'] * (640.0 * 368) / (video_feature['video_height'] * video_feature['video_width']) })
    return video_feature