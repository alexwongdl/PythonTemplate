"""
 @author: AlexWang
 @date: 2020/11/2 3:34 PM
 @Email: alex.wj@alibaba-inc.com
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data_root = '~/temp'
batch_size = 4
num_loader_workers = 2


def load_data():
    transform = transforms.Compose(  # 串联多个图片变换, Compose()类会将transforms列表里面的transform操作进行遍历
        [transforms.ToTensor(),  # 把灰度范围从0-255变换到0-1之间
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 把0-1变换到(-1,1), (0-0.5)/0.5=-1, (1-0.5)/0.5=1

    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_loader_workers)

    testset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_loader_workers)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


class CIFAR_NET(nn.Module):
    def __init__(self):
        super(CIFAR_NET, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.soft = nn.Softmax(dim=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # reshape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.soft(self.fc3(x))
        x = self.fc3(x)
        return x


def train_model():
    # 定义网络
    cifar_net = CIFAR_NET()
    # 定义损失函数
    # \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
    # = -x[class] + \log\left(\sum_j \exp(x[j])\right)
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(cifar_net.parameters(), lr=0.001, momentum=0.9)

    trainloader, testloader, classes = load_data()
    print(type(trainloader))
    for epoch in range(1000):  # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0.0
        total = 0

        for i, data in enumerate(trainloader):
            # inputs:[batch_size, channel, width, height]
            inputs, labels = data
            # gradient清零
            optimizer.zero_grad()
            # forward
            outputs = cifar_net(inputs)
            # accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # update prameters
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print(inputs.shape)
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f, acc:%.3f' %
                      (epoch + 1, i + 1, running_loss / 2000, correct/total))
                running_loss = 0.0
                correct = 0.0
                total = 0
    print('Finished Training')


if __name__ == '__main__':
    train_model()
