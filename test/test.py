"""
Created by hzwangjian1
on 2017-08-04
"""

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

if __name__=="__main__":
    test()
    test_split()
    test_assign()