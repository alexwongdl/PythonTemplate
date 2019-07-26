"""
Created by Alex Wang
On 2018-12-24
"""

import numpy as np


def check_orthogonality(matrix):
    """
    :param matrix: m * n matrix, m >= n 判断是否列向量正交
    :return:
    """
    m, n = matrix.shape
    if m < n:
        return False

    for i in range(n):
        for j in range(i + 1, n):
            if abs(np.dot(matrix[:, i], matrix[:, j])) > 1e-8:
                return False
    return True


def expand_matrix_with_zeros(matrix):
    """
    :param matrix: m x n matrix, m >= n
    :return:
    """
    m, n = matrix.shape
    new_martix = np.zeros(shape=(m, m), dtype=matrix.dtype)
    new_martix[:, 0:n] = matrix
    return new_martix


def expand_to_square_matrix(matrix):
    """
    构建一个m x m 的单位矩阵, 每个列向量和matrix的向量做判断是否正交
    :param matrix: m x n matrix, m >= n
    :return:
    """
    # TODO
    m, n = matrix.shape
    eye_matrix = np.eye(m, m)
    new_martix = np.zeros(shape=(m, m), dtype=np.float)
    new_martix[:, 0:n] = matrix

    next_index = n
    for i in range(m):
        succeed = True
        for j in range(n):
            # TODO 用施密特正交判断是否可以正交化
            if abs(np.dot(matrix[:, j], eye_matrix[:, i])) > 1e-8:
                succeed = False
                break
        if succeed:
            new_martix[:, next_index] = eye_matrix[:, i]

    return new_martix


def schmidt_orthogonality(matrix_org, debug=False):
    """
    b1 = a1, b2 = a2 - kb1, b3 = a3 - k1b1 - k2b2
    :param matrix_org: m x n matrix, m >= n 且满秩
    :return:
    """
    m, n = matrix_org.shape
    if m < n:
        print('error: row num should be greater than column num')
        return None, None

    matrix_ortho = matrix_org.copy()
    matrix_ortho = np.asarray(matrix_ortho, dtype=np.float)
    coefficient = np.zeros(shape=(m, n))
    coefficient[0, 0] = 1

    for i in range(1, n):  # next column
        coefficient[i, i] = 1
        for j in range(i):
            b_j = matrix_ortho[:, j]
            k_j = np.dot(b_j, matrix_org[:, i]) / np.dot(b_j, b_j)
            coefficient[j, i] = k_j

            if debug:
                print(matrix_ortho)
                print('b_{}{}:'.format(i, j).center(10), b_j)
                print('k_{}{}:'.format(i, j).center(10), k_j, '{}x{}/{}x{}'.format(b_j, matrix_org[:, i], b_j, b_j))
                print('{}-={}:\n'.format(matrix_ortho[:, i], k_j * b_j))

            matrix_ortho[:, i] -= k_j * b_j

    for i in range(n):
        devider = np.dot(matrix_ortho[:, i], matrix_ortho[:, i])
        if abs(devider) < 1e-16:  # 0
            matrix_ortho[:, i] *= 0
        else:
            devider = np.sqrt(devider)
            matrix_ortho[:, i] /= devider
            coefficient[i, :] *= devider

    print('result:', matrix_ortho)
    return matrix_ortho, coefficient


def test_orgthogonality():
    arr = np.array([[1, 2, 2],
                    [1, 0, 2],
                    [0, 1, 1]])
    arr_ortho, coefficient = schmidt_orthogonality(arr, True)
    print('coefficient:', coefficient)
    multi_result = np.dot(arr_ortho, coefficient)
    print('multi_result:', multi_result)
    print('-----------------------------------------------')

    arr = np.array([[1, -1, 1],
                    [-1, 1, 1],
                    [0, 1, 1]])
    schmidt_orthogonality(arr, True)
    print('-----------------------------------------------')

    arr = np.array([[1],
                    [-1],
                    [0]])
    schmidt_orthogonality(arr, True)
    print('-----------------------------------------------')

    arr = np.array([[1, -1],
                    [-1, 1],
                    [0, 1]])
    arr_ortho, coefficient = schmidt_orthogonality(arr, True)
    print(expand_matrix_with_zeros(arr))
    print('coefficient:', coefficient)
    print(check_orthogonality(arr))
    print(check_orthogonality(arr_ortho))

    print('expand matrix:', expand_to_square_matrix(arr_ortho))
    print('-----------------------------------------------')

    arr = np.array([[2, 0, 5],
                    [0, 3.3, 2],
                    [0, 0, 0]])
    schmidt_orthogonality(arr, True)


if __name__ == '__main__':
    test_orgthogonality()
