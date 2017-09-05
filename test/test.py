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

if __name__=="__main__":
    test()
    test_split()
    test_assign()
    phog_dim()
    test_md5_hexdigest()