# coding: utf-8
"""
Created by hzwangjian1
on 2017-08-04

图像黑边检测
"""
def row_black_box(image, threshold, diff):
    """
    左右两边的黑边检测
    image: cv.imread读取的图片
    threshold:判断为黑边的条件sum([r,g,b] - [0,0,0](black))
    diff:左右黑边的最大差值
    """
    nRow = image.shape[0]
    nCol = image.shape[1]
    left = -1
    right = nCol
    step = int(nRow/30) # 只检查10行
    for i in range(int(nRow/3), int(nRow/3*2), step):
        curLeft = -1
        curRight = nCol
        for j in range(0,nCol-1):
            if(curLeft == j-1 and sum(list(image[i,j]))<=threshold):
                curLeft += 2
        if left == -1:
            left = curLeft
        if curLeft < left:
            left = curLeft

        for j in range(nCol-1,0,-1):
            if(curRight == j+1 and sum(list(image[i,j]))<=threshold):
                curRight -= 2
        if right == nCol:
            right = curRight
        if curRight > right:
            right = curRight

    if min(left,right)>=1 and abs((left+1)-(nCol-right))<=diff and right-left>0 :
        return left, right
    elif min(left, (nCol-right)) >= 20:
        return min(left, (nCol-right)), min(left, (nCol-right))
    return -1,-1

def col_black_box(image, threshold, diff):
    """
    上下两边的黑边检测
    image: cv.imread读取的图片
    threshold:判断为黑边的条件sum([r,g,b] - [0,0,0](black))
    diff:左右黑边的最大差值
    """
    nRow = image.shape[0]
    nCol = image.shape[1]
    topper = -1
    buttom = nRow
    step = int(nCol/30)
    for i in range(int(nCol/3), int(nCol/3*2), step):
        curTopper = -1
        curButtom = nRow
        for j in range(0,nRow-1):
            if(curTopper == j-1 and sum(list(image[j,i]))<=threshold):
                curTopper += 2
        if topper == -1:
            topper = curTopper
        if curTopper < topper:
            topper = curTopper

        for j in range(nRow-1,0,-1):
            if(curButtom == j+1 and sum(list(image[j,i]))<=threshold):
                curButtom -= 2
        if buttom == nRow:
            buttom = curButtom
        if curButtom > buttom:
            buttom = curButtom

    if min(topper,buttom)>=1 and abs((topper+1)-(nRow-buttom))<=diff and buttom-topper>0 :
        return topper, buttom
    elif min(topper,(nRow-buttom))>=20:
        return min(topper,(nRow-buttom)), min(topper,(nRow-buttom))

    return -1,-1