"""
Created by Alex Wang
On 2018-08-26
"""
import struct
import base64
import math
import traceback

import numpy as np

TAG_NUM = 1746
AUDIO_FEAT_LEN = 400
AUDIO_FEAT_DIM = 128

FRAME_FEAT_LEN = 200
FRAME_FEAT_DIM = 1024


def tags_process(tags_org):
    """
    :param tags_org:
    | 782028346  | 1076,1676,1373,0,0,0,0,0,0,0 |
    | 825916601  | 515,1676,0,0,0,0,0,0,0,0 |
    | 702112086  | 1123,451,1325,1326,0,0,0,0,0,0 |
    | 798679449  | 105,1173,630,1089,0,0,0,0,0,0 |
    | 399612636  | 1089,1123,0,0,0,0,0,0,0,0 |
    tag:1-1746
    :return:
    """
    new_tags = []
    for tag in tags_org:
        elems = tag.strip().split(',')
        tag_ids = [int(elem) for elem in elems if int(elem) != 0]
        tag_vec = np.zeros(shape=(TAG_NUM), dtype=np.int32)
        for tag in tag_ids:
            tag_vec[tag - 1] = 1
        new_tags.append(tag_vec)
    return np.asarray(new_tags)


def decode_feature(feature, n, fmt="f"):
    result = struct.unpack(fmt * n, base64.urlsafe_b64decode(feature))
    return list(result)


def audio_feat_process(audio_feat_org):
    """
    :param audio_feat_org:
    :return:[batch_size, 400, 128]
    """
    audio_feat_new = []
    for audio_feat in audio_feat_org:
        audio_features = map(lambda f: decode_feature(f, AUDIO_FEAT_DIM, fmt='Q'),
                             [item.strip() for item in audio_feat.split("\t") if len(item.strip()) == 1368])
        len_fea = len(audio_features)
        new_arr = np.zeros(shape=(AUDIO_FEAT_LEN, AUDIO_FEAT_DIM))
        new_arr[0:len_fea, :] = np.asarray(audio_features)

        # print('shape of audio_features:{}'.format(np.asarray(new_arr).shape)) # (400, 128)
        audio_feat_new.append(new_arr)
    return np.array(audio_feat_new)


def frame_feat_process(frame_feat_org):
    """
    :param frame_feat_org:
    :return:[batch_size, 200, 1024]
    """
    frame_feat_new = []
    for frame_feat in frame_feat_org:
        rgb_features = map(lambda f: decode_feature(f, 1024, fmt='f'),
                           [item.strip() for item in frame_feat.split("\t")
                            if len(item.strip()) == 5464])

        len_fea = len(rgb_features)
        new_arr = np.zeros(shape=(FRAME_FEAT_LEN, FRAME_FEAT_DIM))
        new_arr[0:len_fea, :] = np.asarray(rgb_features)

        frame_feat_new.append(new_arr)
    return np.array(frame_feat_new)


def image_feat_process(image_feat_org):
    """
    :param image_feat_org:
    :return: [batch_size, 2048]
    """
    image_feat_new = []
    for image_feat in image_feat_org:
        img_new = decode_feature(image_feat, 2048)
        image_feat_new.append(np.asarray(img_new))
    return np.array(image_feat_new)

def frame_feat_process_lstm(frame_feat_org):
    """
    :param frame_feat_org:
    :return:[batch_size, 200, 1024]
    """
    # layer 2 (batch * (198/4-4) * 256) = (batch * 45 * 256)
    x_length = np.ones(dtype=np.int32, shape=(len(frame_feat_org))) * 45
    frame_feat_new = []
    for idx, frame_feat in enumerate(frame_feat_org):
        rgb_features = map(lambda f: decode_feature(f, 1024, fmt='f'),
                           [item.strip() for item in frame_feat.split("\t")
                            if len(item.strip()) == 5464])

        len_fea = len(rgb_features)
        lstm_len = min(math.ceil(len_fea/200.0 * 45), 45)
        x_length[idx] = lstm_len

        new_arr = np.zeros(shape=(FRAME_FEAT_LEN, FRAME_FEAT_DIM))
        new_arr[0:len_fea, :] = np.asarray(rgb_features)

        frame_feat_new.append(new_arr)
    return np.array(frame_feat_new), x_length


def text_feat_propress(text_feat_org):
    """
    :param text_feat_org:
    :return:
    """
    text_feat_new = []
    for text_feat in text_feat_org:
        elems = text_feat.split(',')
        elems_int = [int(elem) for elem in elems]
        text_feat_new.append(elems_int)

    return np.array(text_feat_new)


if __name__ == '__main__':
    print(tags_process(['1076,1676,1373,1,0,0,0,0,0,0', '515,1676,0,1746,0,0,0,0,0,0']))
    text_fea = text_feat_propress(['125880,205691,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0', '125880,205691,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0'])

    print(text_fea)
    print(text_fea.shape)