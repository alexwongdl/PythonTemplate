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



if __name__ == "__main__":
    test_slice_sequence()

