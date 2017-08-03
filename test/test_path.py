"""
Created by Alex Wang
on 2017-08-03
"""
import os


def test_basename():
    """
    获取路径的最后一部分
    :return:
    """
    print(os.path.basename('http://flv3.bn.netease.com/videolib3/1707/24/HhsvJ4943/HD/HhsvJ4943-mobile.mp4'))


def iterate_path():
    """
    递归遍历文件夹 path:路径；dirnames：路径下的目录；filenames：路径下的非目录
    :return:
    """
    for path, dirnames, filenames in os.walk('E://temp/videoquality'):
        print("path is :" + path)
        for dirname in dirnames:
            print("dirname:" + dirname)
        for filename in filenames:
            print("filename:" + filename)


def test_list_files():
    """
    列出当前目录下的所有文件，用isfile和isdir来筛选文件或者目录
    :return:
    """
    root = 'E://temp/videoquality'
    for file in os.listdir(root):
        if os.path.isfile(os.path.join(root,file)):
            print(file)


if __name__ == "__main__":
    # test_basename()
    # iterate_path()
    test_list_files()
