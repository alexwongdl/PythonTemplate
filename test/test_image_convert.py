"""
Created by Alex Wang
On 2017-10-10
"""
import os
from PIL import Image

from myutil import pathutil

def jpg2png():
    root_path = 'E://temp/videoquality/heibian_keyframe-small'
    file_obs_list, file_list = pathutil.list_files(root_path)
    for (abs_path, file_name) in zip(file_obs_list, file_list):
        img=Image.open(abs_path)
        img.save(os.path.join(root_path, file_name.split('.')[0] + '.png'))


if __name__ == "__main__":
    jpg2png()