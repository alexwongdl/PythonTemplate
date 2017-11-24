"""
Created by Alex Wang on 2017-06-21
测试进度条显示模块tqdm
"""

from tqdm import tqdm
from tqdm import trange
import time

def test():
    test_tqdm()

def test_tqdm():
    ### 最简单格式
    for i in tqdm(range(300),desc="1st progress"):
        time.sleep(0.01)

    for i in trange(100):
        time.sleep(0.01)

    ### 通用格式  total/update/description
    pbar = tqdm(range(300))
    for i in range(30):
        pbar.set_description("{} has been processed".format(i))
        pbar.update(10)
        time.sleep(0.1)
    pbar.close()


if __name__ == '__main__':
    test()