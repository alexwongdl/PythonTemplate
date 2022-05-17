"""
torchvision图像分类：[https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

[https://blog.csdn.net/weixin_43917574/article/details/114625616](https://blog.csdn.net/weixin_43917574/article/details/114625616)

****torchvision.datasets.ImageFolder：****

[https://blog.csdn.net/taylorman/article/details/118631209](https://blog.csdn.net/taylorman/article/details/118631209)
"""
import os
import time
from sklearn.metrics import confusion_matrix
import random

from matplotlib import font_manager
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from warmup_scheduler import GradualWarmupScheduler

DATA_ROOT = "./"

train_dir = os.path.join(DATA_ROOT, "train")
test_dir = os.path.join(DATA_ROOT, "valid")

# 将图像调整为224×224尺寸并归一化
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
train_set = datasets.ImageFolder(train_dir, transform=train_augs)
test_set = datasets.ImageFolder(test_dir, transform=test_augs)
print(train_set.classes)
print(train_set.class_to_idx)

batch_size = 128
train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=64)
test_iter = DataLoader(test_set, batch_size=batch_size, num_workers=32)


def denorm(img):
    for i in range(img.shape[0]):
        img[i] = img[i] * std[i] + mean[i]
    return img


# plt.figure(figsize=(8, 8))
# for i in range(9):
#     img, label = train_set[random.randint(0, len(train_set))]
#     # label = train_set.classes[label]
#     img = denorm(img)
#     img = img.permute(1, 2, 0)
#
#     fontP = font_manager.FontProperties()
#     print(fontP.get_family())
#     fontP.set_family('sans-serif')
#     fontP.set_size(14)
#
#     ax = plt.subplot(3, 3, i + 1)
#     ax.imshow(img.numpy())
#     ax.set_title("label = {}".format(label), fontproperties=fontP)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:{}".format(device))


def train(net, train_iter, test_iter, criterion, optimizer, scheduler_warmup, num_epochs):
    net = net.to(device)
    best_acc = 0

    print("training on", device)
    for epoch in range(num_epochs):

        start = time.time()
        net.train()  # 训练模式
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()  # 梯度清零
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        l2_reg = torch.tensor(0.).to(device)
        for param in net.parameters():
            l2_reg += torch.norm(param)

        y_true = []
        y_pred = []
        with torch.no_grad():
            net.eval()  # 评估模式
            test_acc_sum, n2 = 0.0, 0
            for X, y in test_iter:
                predict = net(X.to(device)).argmax(dim=1)
                test_acc_sum += (predict == y.to(device)).float().sum().cpu().item()
                n2 += y.shape[0]

                y_true.extend(y)
                y_pred.extend(predict.tolist())

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec, l2 reg %.4f, lr %.8f'
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc_sum / n2,
                 time.time() - start, l2_reg.cpu().item(), optimizer.param_groups[0]['lr']))
        scheduler_warmup.step(epoch)

        valid_acc = test_acc_sum / n2
        if valid_acc > (best_acc - 0.03):
            if valid_acc > best_acc and valid_acc > 0.7:
                best_acc = valid_acc
                torch.save(net, os.path.join(DATA_ROOT, "{}_{}.pth".format(epoch, best_acc)))
            print(confusion_matrix(y_true, y_pred))


pretrained_net = models.resnet50(pretrained=True)
num_ftrs = pretrained_net.fc.in_features
pretrained_net.fc = nn.Linear(num_ftrs, len(train_set.classes))

output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
# lr = 0.0001
lr = 0.0005
# optimizer = optim.SGD([{'params': feature_params},
#                        {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
#                       lr=lr, weight_decay=0.001)

optimizer = optim.Adam([{'params': feature_params},
                        {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001)

loss = torch.nn.CrossEntropyLoss()
scheduler_steplr = StepLR(optimizer, step_size=1, gamma=0.95)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_steplr)

train(pretrained_net, train_iter, test_iter, loss, optimizer, scheduler_warmup, num_epochs=500)