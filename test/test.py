"""
Created by hzwangjian1
on 2017-08-04
"""
import hashlib
def test():
    start = 0
    end  = 33
    step = int(end/10)
    aa = [i * 10 for i in range(step)]
    for i in aa:
        print(i)

    for j in range(0, 368, 10):
        print(j)

def test_split():
    category = '游戏/直播'
    root_category = category.split('/')[0]
    print(root_category)

def test_assign():
    org = 5
    a,b,c = [org] * 3
    print(a)
    print(c)

def phog_dim():
    with open('E://temp/docduplicate/recallOpt/temp.txt','r') as rhandler:
        line = rhandler.readline()
        sub_strs = line.split(',')
        print(len(sub_strs))

def test_md5_hexdigest():
    signature = hashlib.md5(('0c8dd438-1e78-428f-b4c9-42cf41c13dcb' + "2017-09-01 12:00:00").encode()).hexdigest()
    print(signature)

def list_rerange(list_org):
    lennth = len(list_org)
    times_4 = []
    times_2 = []
    times_1 = []
    for item in list_org:
        if item % 4 == 0:
            times_4.append(item)
        elif item % 2 == 0:
            times_2.append(item)
        else:
            times_1.append(item)
    print((times_4))
    print((times_2))
    print((times_1))

    len_times_4 = len(times_4)
    len_times_2 = len(times_2)
    len_times_1 = len(times_1)

    if len_times_2 == 0 and len_times_1 == 0: #1 和 2都没有
        return True

    if len_times_2 == 0: ## 没有2有1
        if len_times_4 * 2 >= len_times_1: ## 4的个数是1的个数的两倍
            return True
        else :
            return False
    if len_times_1 == 0 : ## 没有1有2
        if len_times_2 % 2 == 0: ## 2个2
            return True
        else: ## 1个2
            if len_times_4 > 0:
                return True
            else:
                return False

    ## 1和2都有
    if len_times_2 % 2 == 0: ## 2的个数是偶数
        if len_times_4 * 2 > len_times_1: ## 4的个数是1的个数的两倍
            return True
        else :
            return False
    else:  ## 2的个数是奇数,相同
        if len_times_4 * 2 > len_times_1:## 4的个数是1的个数的两倍
            return True
        else :
            return False

import scipy.misc as sic
import numpy as np
def test_png():
    im = sic.imread("E://temp/videoquality/heibian_keyframe_small/VAPS2EQ4T_544_960_KF_012.png")
    im = sic.imread("E://temp/deblur/images/r5e7ce087t.png")
    print(im.shape)
    print(im[50][:][:])

from myutil import httputil
import os
def load_imgs():
    output_dir = "E://temp/deblur/imags_dir"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open('E://temp/deblur/video_article_pic', 'r') as fin:
        line = fin.readline()
        print(line)
        while line:
            substrs = line.split("\t")
            docid, url = substrs[0], substrs[1]
            save_path = os.path.join(output_dir, docid + '.jpg')
            httputil.image_download(url, save_path)
            line = fin.readline()

if __name__=="__main__":
    # test()
    # test_split()
    # test_assign()
    # phog_dim()
    # test_md5_hexdigest()
    print(list_rerange([8])) #没有1和2
    print("===============================")
    print(list_rerange([2,3,5,7,8,6]))  #有1和2
    print("===============================")
    print(list_rerange([3,5,8])) #没有2
    print("===============================")
    print(list_rerange([2])) #没有1
    print("===============================")
    print(list_rerange([2,3,8,6]))  #有1和2
    print("===============================")
    print(list_rerange([1,4,1,4,1,4,1]))  #有1和2
    print("===============================")
    print(list_rerange([]))

    digit_list = [1,2,7,4,5]
    del digit_list[3]
    print(digit_list)
    print(digit_list[1:3])
    a,b = [],[]
    a.append(1)
    print('a:')
    print(a)
    a.extend([0]*3)
    print(a)
    a.extend([0]*0)
    print(a)

    print('sum a:')
    a = [[1,2,3],[11,12,13]]
    b = sum(a[0])
    print(b)

    print('---------------test dict---------------------')
    dict_a = {'a':1, 'b':2, 'c':3}
    if 'a' in dict_a.keys():
        print('dict_a[a]:{}'.format(dict_a['a']))
    # load_imgs()