import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
import torch.optim.lr_scheduler
import os
import copy
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from layers import ConvOffset2D
from combine import SplitCGCconv


# CGCconv
class CConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(CConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
        # for convolutional layers with a kernel size of 1, just use traditional convolution
        if kernel_size == 1 :
            self.ind = True
        else:
            self.ind = False
            self.oc = out_channels
            self.ks = kernel_size

            # the target spatial size of the pooling layer
            ws = kernel_size
            self.avg_pool = nn.AdaptiveAvgPool2d((ws, ws))

            # the dimension of the latent repsentation
            self.num_lat = int((kernel_size * kernel_size) / 2 + 1)

            # the context encoding module
            self.ce = nn.Linear(ws * ws, self.num_lat, False)
            self.ce_bn = nn.BatchNorm1d(in_channels)
            self.ci_bn2 = nn.BatchNorm1d(in_channels)

            # activation function is relu
            self.act = nn.ReLU(inplace=True)

            # the number of groups in the channel interacting module
            if in_channels // 16:
                self.g = 16
            else:
                self.g = in_channels
            # the channel interacting module
            self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
            self.ci_bn = nn.BatchNorm1d(out_channels)

            # the gate decoding module
            self.gd = nn.Linear(self.num_lat, kernel_size * kernel_size, False)
            self.gd2 = nn.Linear(self.num_lat, kernel_size * kernel_size, False)

            # used to prrepare the input feature map to patches
            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

            # sigmoid function
            self.sig = nn.Sigmoid()

    def forward(self, x):
        # for convolutional layers with a kernel size of 1, just use traditional convolution
        if self.ind:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            b, c, h, w = x.size()
            weight = self.weight
            # allocate glbal information
            gl = self.avg_pool(x).view(b, c, -1)
            # context-encoding module
            out = self.ce(gl)
            # use different bn for the following two branches
            ce2 = out
            out = self.ce_bn(out)
            out = self.act(out)
            # gate decoding branch 1
            out = self.gd(out)
            # channel interacting module
            if self.g > 3:
                # grouped linear
                oc = self.ci(self.act(self.ci_bn2(ce2). \
                                      view(b, c // self.g, self.g, -1).transpose(2, 3))).transpose(2, 3).contiguous()
            else:
                # linear layer for resnet.conv1
                oc = self.ci(self.act(self.ci_bn2(ce2).transpose(2, 1))).transpose(2, 1).contiguous()
            oc = oc.view(b, self.oc, -1)
            oc = self.ci_bn(oc)
            oc = self.act(oc)
            # gate decoding branch 2
            oc = self.gd2(oc)
            # produce gate
            out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.oc, 1, self.ks, self.ks))
            # unfolding input feature map to patches
            x_un = self.unfold(x)
            b, _, l = x_un.size()
            # gating
            out = (out * weight.unsqueeze(0)).view(b, self.oc, -1)
            # currently only handle square input and output
            return torch.matmul(out, x_un).view(b, self.oc, int(np.sqrt(l)), int(np.sqrt(l)))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 context gated convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def cconv3x3(in_planes, out_planes, stride=1):
    """3x3 context gated convolution with padding"""
    return CConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def cconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return CConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# splitCGC+Dconv
class SplitBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SplitBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SplitCGCconv(planes, planes, 3,stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SplitCGCconv(planes, planes * self.expansion,1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# CGC、CGC+Dconv
class CGCBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CGCBottleneck, self).__init__()
        self.conv1 = cconv1x1(inplanes, planes)
        self.offset1 = ConvOffset2D(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = cconv3x3(planes, planes, stride)
        self.offset2 = ConvOffset2D(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = cconv1x1(planes, planes * self.expansion)
        self.offset3 = ConvOffset2D(planes)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.offset2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # out = self.offset3(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=9, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                cconv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, my=False, ft=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
            # 数据标签转换为Tensor
        return img, label

    def __len__(self):
        return len(self.imgs)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = ResNet(SplitBottleneck, [3, 4, 6, 3])
net.to(device)
print(net)

transforms = transform.Compose([
    #transform.Grayscale(),
    transform.Resize([320,320]),
    #transform.RandomHorizontalFlip(),  # 随机水平翻转，概率为0.5
    #transform.RandomVerticalFlip(),
    transform.ToTensor()  # 转化数据类型为tensor
])
transforms_test = transform.Compose([
    #transform.Grayscale(),
    transform.Resize([320,320]),
    transform.ToTensor()
])
train_data = MyDataset(txt='C:/Users/cyw/Desktop/c20201105/train.txt', transform=transforms)
test_data = MyDataset(txt='C:/Users/cyw/Desktop/c20201105/test.txt', transform=transforms_test)
# 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False)
print('num_of_trainData:', len(train_data))
print('num_of_testData:', len(test_data))


epochs = 120
lr = 0.00025
batch_size = 2
test_acc = []
train_acc = []

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr,momentum=0.95,weight_decay=5e-4)


# 用于更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 训练
total_step = len(train_loader)
curr_lr = lr
bets_acc = 0.0
bets_epoch = 0


name = []
milestones = [5]
wrong = open('C:/Users/cyw/Desktop/wrong2.txt', 'r+')
classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8')

for epoch in range(epochs):
    total = 0
    num_correct = 0
    for i, (images, labels) in enumerate(train_loader):
        # 将数据放入GPU
        net.train()

        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        num_correct += (predicted == labels).sum().item()
        total += labels.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 90 == 0:
            print("Epoch [{}/{}],step[{}/{}] Loss:{:.4f} Acc:{:.4f} {}/{}"
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item(),num_correct/len(train_data),num_correct,len(train_data)))
    train_acc.append(num_correct/len(train_data))
    print('epoch:{} train-acc:{} correct{} total{}'.format(epoch+1,num_correct/total,num_correct,total))

    # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    # 降低学习速度
    if (epoch + 1) % 40 == 0 :  # 每过20个Epoch，学习率就会下降
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
    class_correct = list(0. for i in range(12))
    class_total = list(0. for i in range(12))
# 测试

    name.clear()
    net.eval()
    with torch.no_grad():
        # correct = 0
        # total = 0
        # truth_class = 0
        num_correct = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            num_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()

            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                if predicted[i]!=labels[i]:

                    #name = str(int(predicted[i])) + str(int(labels[i])) + '\n'
                    name.append((str(int(predicted[i])),str(int(labels[i]))))

        if num_correct / len(test_data) > bets_acc:
            wrong.seek(0)
            wrong.truncate(0)
            bets_acc = num_correct / len(test_data)
            bets_epoch = epoch + 1
            classc = copy.deepcopy(class_correct)
            classt = copy.deepcopy(class_total)
            for x1,x2 in name:
                x = str(x1)+str(x2)+'\n'
                wrong.write(x)

           # torch.save(net,'net-{}'.format(epoch))
    print('acc:{:4f} {}/{}'.format(num_correct/len(test_data),num_correct,len(test_data)))
    test_acc.append(num_correct/len(test_data))
print('best_acc:{}   best_epoch:{}'.format(bets_acc,bets_epoch))
try:
    for i in range(9):
        print('Accuracy of %5s : %2d %% %2d/%2d ' % (
            classes[i], 100 * classc[i] / classt[i],classc[i], classt[i]))
except NameError:
    for i in range(9):
        print('Accuracy of %5s : %2d %% %2d/%2d ' % (
            classes[i], 100 * class_correct[i] / class_total[i],class_correct[i], class_total[i]))
wrong.close()
x1 = range(1, epochs+1)
x2 = range(1, epochs+1)
y1 = test_acc
y2 = train_acc
plt.plot(x1, y1, 'o-',label='test_acc')
plt.plot(x2, y2, '.-',label='train_acc')
plt.legend(loc = 'upper right')
plt.xlabel('acc')
plt.ylabel('epoch')
plt.xticks(range(epochs))
plt.show()