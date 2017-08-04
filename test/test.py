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

if __name__=="__main__":
    test()