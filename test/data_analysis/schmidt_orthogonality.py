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
            if np.dot(matrix[:, i], matrix[:, j]) != 0:
                return False
    return True


def expand_matrix(matrix):
    """
    :param matrix: m x n matrix, m >= n
    :return:
    """
    m, n = matrix.shape
    new_martix = np.zeros(shape=(m, m), dtype=matrix.dtype)
    new_martix[:, 0:n] = matrix
    return new_martix


def schmidt_orthogonality(matrix_org, debug=False):
    """
    b1 = a1, b2 = a2 - kb1, b3 = a3 - k1b1 - k2b2
    :param matrix_org: m x n matrix, m >= n 且满秩
    :return:
    """
    m, n = matrix_org.shape
    if m < n:
        print('error: input matrix should be full rank')
        return None

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

    print('result:', matrix_ortho)
    return matrix_ortho, coefficient


if __name__ == '__main__':
    # arr = np.array([[1, 2, 2],
    #                 [1, 0, 2],
    #                 [0, 1, 1]])
    # schmidt_orthogonality(arr, True)
    #
    # arr = np.array([[1, -1, 1],
    #                 [-1, 1, 1],
    #                 [0, 1, 1]])
    # schmidt_orthogonality(arr, True)
    #
    # arr = np.array([[1],
    #                 [-1],
    #                 [0]])
    # schmidt_orthogonality(arr, True)

    arr = np.array([[1, -1],
                    [-1, 1],
                    [0, 1]])
    arr_ortho, coefficient = schmidt_orthogonality(arr, True)
    print(expand_matrix(arr))
    print('coefficient:', coefficient)
    print(check_orthogonality(arr))
    print(check_orthogonality(arr_ortho))

