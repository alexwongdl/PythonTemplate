"""
Created by Alex Wang
on 2017-08-03
"""
import os
import hashlib

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

def test_file_exist():
    root = 'E://temp/videoquality/abc'
    if os.path.isdir(root):
        print('file exists')
    else:
        print('file not exists')

def test_delete():
    """
    os.rmdir：删除空目录，目录不为空时抛出异常
    os.remove：删除文件，如果是一个目录，抛出异常
    :return:
    """
    dir_path = 'E://temp/videoquality/test'
    os.rmdir(dir_path)

    file_path = 'E://temp/videoquality/test.txt'
    os.remove(file_path)


def current_dir():
    print(os.getcwd()) ##当前目录
    print(os.path.abspath(__file__)) ##当前文件
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  ##当前目录的上一级
    print(cwd)

def dir_clear(dir_path):
    """
    清空一个目录
    :param dir_path:
    :return:
    """
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path,file)):
            os.remove(os.path.join(dir_path,file))



if __name__ == "__main__":
    test_basename()
    # iterate_path()
    # test_list_files()
    # test_delete()
    current_dir()
    # test_file_exist()
