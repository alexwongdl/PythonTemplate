"""
Created by Alex Wang
On 2017-11-11
数学计算单元
"""

def digit_round(value, num):
    """
    value小数点后面保留num位有效数字，最后一位四舍五入
    :param value:
    :param num:
    :return:
    """
    return round(value, num)

def median(digit_list):
    """
    返回digit_list的中位数
    :param digit_list:
    :return:
    """
    data = sorted(digit_list)
    half = len(data) // 2
    return (data[half] + data[~half]) / 2

if __name__ == '__main__':
    print("test")
    print(digit_round(152.345435, 3))

    print(median([6,3,8,2,5]))