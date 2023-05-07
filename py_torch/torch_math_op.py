import torch


def tensor_product():
    """
    :return:
    """
    #  逐元素乘 element-wise product
    print("-"*20 + "element-wise" + "-"*20)
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = torch.tensor([10])  # broadcast
    print(torch.mul(a, b), a * b, a * c)

    # 向量点乘
    print("-"*20 + "dot" + "-"*20)
    print(torch.matmul(a, b), torch.dot(a, b))

    # 矩阵相乘
    print("-"*20 + "matrix multiple" + "-"*20)
    ma = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mb = torch.tensor([[7, 8], [9, 10], [11, 12]])
    print(torch.matmul(ma, mb), ma@mb,  torch.mm(ma, mb), torch.tensordot(ma, mb, ([1], [0])))

    # tensordot: 对应维度点乘
    print("-"*20 + "tensor dot" + "-"*20)
    a = torch.arange(60.).reshape(3, 4, 5)
    b = torch.arange(24.).reshape(4, 3, 2)
    print(torch.tensordot(a, b, dims=([1, 0], [0, 1])).shape)  # [5, 2]

    a = torch.randn(3, 4, 5)
    b = torch.randn(4, 5, 6)
    print(torch.tensordot(a, b, dims=2))
    print("einsum", torch.einsum("ijk,jkl->il", a, b))
    b = torch.randn(5, 2, 6)
    print(torch.tensordot(a, b, dims=1).shape)
    print("einsum", torch.einsum("ijk,khl->ijhl", a, b).shape)

    # einsum
    print("-"*20 + "einsum" + "-"*20)
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[4, 5, 6], [7, 8, 9]])
    print(torch.einsum("ij,jk->ik", a, b))
    print(torch.einsum("ii->i", a))
    print(torch.einsum("ij->i", a))


if __name__ == '__main__':
    tensor_product()
