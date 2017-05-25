'''
Created on 2017-05-24
@author: Alex Wang
'''


def gen_iter():
    for i in range(10):
        if (i % 2) == 0:
            yield i
        else:
            yield 0


def test():
    iter = gen_iter()
    for i in iter:
        print(i)
    iter_list = list(iter)
    print(iter_list)  ##generator只能被调用一次

    iter_list = list(gen_iter())  ## generator转换为list
    print(iter_list)

if __name__ == "__main__":
    test()
