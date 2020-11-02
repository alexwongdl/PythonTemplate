from __future__ import print_function
import numpy as np
import torch


def test_tensor():
    a = torch.empty(5, 3)
    b = torch.rand(5, 3)
    c = torch.zeros(5, 3, dtype=torch.long)
    d = torch.tensor([1.0, 2])
    e = a.new_ones(2, 3, dtype=torch.long)  # 继承了a的属性
    f = torch.randn_like(e, dtype=torch.double)
    g = torch.randn_like(e, dtype=torch.double)

    print(a.size())  # torch.Size([5, 3]), torch.Size的返回值是tuple,支持tuple的所有操作
    print(tuple(a.size()))
    a_reshape = a.view(15)
    print(a_reshape.size())

    # add
    print('-------------------add-----------------')
    print(torch.add(f, g))
    result = torch.empty(2, 3, dtype=torch.double)
    torch.add(f, g, out=result)
    f.add_(g)

    # numpy
    print('-------------------numpy-----------------')
    a_numpy = a.numpy()  # Torch Tensor与NumPy数组共享底层内存地址，修改一个会导致另一个的变化
    print(a_numpy)
    a.add_(1)
    print(a_numpy)
    a_tensor = torch.from_numpy(a_numpy)
    print(a_tensor)

    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)


def test_grad():
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    y = x + 2
    z = y * y * 3
    out = z.mean()
    out.backward()
    print(x.grad)
    print(y.grad)
    print(z.grad)

if __name__ == '__main__':
    # test_tensor()
    test_grad()