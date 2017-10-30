"""
Created by Alex Wang
On 2017-10-30
"""

from myutil import cmdutil
from myutil import pathutil

if __name__ == '__main__':
    dir = '/data/hzwangjian1/image_enhance/heibian_keyframe'
    # dir = 'E://temp/videoDuplicate/img_1030'
    file_obs_list, _ = pathutil.list_files(dir)
    for filename in file_obs_list:
        if filename.endswith('jpg'):
            print(filename)
            cmdutil.run(['python', 'enhance.py', '--type=photo', '--model=deblur', '--zoom=1', filename], timeout=100)
