"""
Created by Alex Wang on 2019-02-22

pip install fs
"""
import time

import cv2
from fs.memoryfs import MemoryFS


def test_memoryfs_video():
    """
    example cannot run
    :return:
    """

    class stuff:
        mem = MemoryFS()
        output = mem.createfile('output.avi')
        rectime = 0
        delay = 0
        kill = 0
        cap = cv2.VideoCapture(0)
        # out = cv2.VideoWriter('C:\motion\\output.avi',cv2.cv.CV_FOURCC('F','M','P','4'), 30, (640,480),True)
        out = cv2.VideoWriter(output, cv2.cv.CV_FOURCC('F', 'M', 'P', '4'), 30, (640, 480), True)

    print("saving")
    movement = time.time()
    while time.time() < int(movement) + stuff.rectime:
        stuff.out.write(frame)
        ret, frame = stuff.cap.read()


def test_memoryfs_text():
    """
    write and read text file
    :return:
    """
    mem = MemoryFS()
    data_path = 'a.txt'
    for i in range(10):
        mem.appendtext(data_path, str(i))

    for line in mem.readtext(data_path):
        print('read:{}'.format(line.strip()))


if __name__ == '__main__':
    test_memoryfs_text()
