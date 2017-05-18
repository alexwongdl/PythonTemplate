'''
Created on 2017-05-12
@author:Alex Wang
'''


def test_slice_sequence():
    '''
    list切片测试
    :return:
    '''
    mylist = ["a","b","c","d","e","f","g","h","i","j"]
    print(mylist[2:8])
    print(mylist[-3:])

    print(mylist[::2])
    print(mylist[::-2])
    print(mylist[1::-1])

    list_gene = (x*2 for x in mylist)
    print(list_gene.__next__())
    print(list_gene.__next__())

    gene_a = (i ** 2 for i in range(10) if i%2==0)
    print(gene_a.__next__())
    print(gene_a.__next__())
    print(next(gene_a))

special_data = [21,6,3]
def toTuple(x):
    if(x in special_data):
        return (0,x)
    else:
        return (1,x)

def test_sort():
    mylist = [4,2,6,21,67,2,4]
    mylist.sort(key=toTuple)
    print(mylist)

    sort_list = sorted(mylist)  # sorted 可以应用于任意迭代数据结构，返回新的数据
    print(sort_list)

    mylist.sort() # 在原始list上排序
    print(mylist)

def test_sort1():
    mylist = [4,2,6,21,67,2,4]
    my_tuple = [toTuple(x) for x in mylist]
    sorted_tuple = sorted(my_tuple)
    x,y = zip(*sorted_tuple)
    print(x)
    print(y)

if __name__ == "__main__":
    # test_slice_sequence()
    test_sort()
    test_sort1()
