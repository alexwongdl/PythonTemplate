"""
Created by Alex Wang
On 2017-10-10
"""
import os
from PIL import Image
import tensorflow as tf
import scipy.misc as sim

from myutil import pathutil

def jpg2png_one(input_path, output_path):
    img = sim.imread(input_path)
    sim.imsave(output_path, img)

def jpg2png():
    # sess = tf.Session()
    org_path = 'E://temp/videoquality/heibian_keyframe_small_jpg'
    root_path = 'E://temp/videoquality/heibian_keyframe_small'
    file_obs_list, file_list = pathutil.list_files(org_path)
    for (abs_path, file_name) in zip(file_obs_list, file_list):
        # img=Image.open(abs_path)
        # img = img.convert('PNG')
        # img.save(os.path.join(root_path, file_name.split('.')[0] + '.png'), 'PNG')

        # img = sim.imread(abs_path)
        # sim.imsave(os.path.join(root_path, file_name.split('.')[0] + '.png'), img)
        jpg2png(abs_path, os.path.join(root_path, file_name.split('.')[0] + '.png'))

if __name__ == "__main__":
    # jpg2png()
    jpg2png_one('E://temp/deblur/images/r5fd51a76t.jpg','E://temp/deblur/images/r5fd51a76t.png')