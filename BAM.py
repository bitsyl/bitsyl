import torch
import torch.nn as nn
import torchvision
import math
from torch.nn import init

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transform
import torch.optim.lr_scheduler
import os
import copy
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
    def forward(self, x):
        avg = self.avg_pool(x).view(x.size(0), -1)
        out = self.shared_MLP(avg).unsqueeze(2).unsqueeze(3).expand_as(x)
        return out
class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, dilation_conv_num=2, dilation_rate=4):
        super(SpatialAttention, self).__init__()
        mid_channel = channel // reduction
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(channel, mid_channel, kernel_size=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )
        dilation_convs_list = []
        for i in range(dilation_conv_num):
            dilation_convs_list.append(nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=dilation_rate, dilation=dilation_rate))
            dilation_convs_list.append(nn.BatchNorm2d(mid_channel))
            dilation_convs_list.append(nn.ReLU(inplace=True))
        self.dilation_convs = nn.Sequential(*dilation_convs_list)
        self.final_conv = nn.Conv2d(mid_channel, channel, kernel_size=1)
    def forward(self, x):
        y = self.reduce_conv(x)
        x = self.dilation_convs(y)
        out = self.final_conv(y)#.expand_as(x)
        return out
class BAM(nn.Module):
    """
        BAM: Bottleneck Attention Module
        https://arxiv.org/pdf/1807.06514.pdf
    """
    def __init__(self, channel):
        super(BAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention(channel)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        att = 1 + self.sigmoid(self.channel_attention(x) * self.spatial_attention(x))
        return att * x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,inplanes,outplanes,stride=1,downsample=None,norm_layer=None,padding=1,):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes,outplanes)
        self.conv2 = conv3x3(outplanes,outplanes,stride,padding)
        self.conv3 = conv1x1(outplanes,outplanes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.bn3 = nn.BatchNorm2d(outplanes*self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.bam = BAM(outplanes*self.expansion)

    def forward(self,x):
        identity = x


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)


        out = self.conv3(out)
        out = self.bn3(out)
        out = self.bam(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck1(nn.Module):
    expansion = 4

    def __init__(self,inplanes,outplanes,stride=1,downsample=None,norm_layer=None,padding=1,):
        super(Bottleneck1, self).__init__()
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes,outplanes)
        self.conv2 = conv3x3(outplanes,outplanes,stride,padding)
        self.conv3 = conv1x1(outplanes,outplanes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.bn3 = nn.BatchNorm2d(outplanes*self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.bam = BAM(outplanes*self.expansion)

    def forward(self,x):
        identity = x


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)


        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,se_block,block,layers,num_classes=9,norm_layer=None ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,)
        self.layer4 = self._make_layer(se_block, 512, layers[3], stride=2,)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(

                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                             norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def conv3x3(inplanes,outplanes,stride=1,padding=1):
    return nn.Conv2d(inplanes,outplanes,kernel_size=3,stride=stride,padding=padding)


def conv1x1(inplanes,outplaens,stride=1):
    return nn.Conv2d(inplanes,outplaens,kernel_size=1,stride=stride,bias=False)


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


transforms = transform.Compose([
    #transform.Grayscale(),
    transform.Resize([320,320]),
    # transforms.RandomResizedCrop([224,224]),
    transform.RandomHorizontalFlip(),  # 随机水平翻转，概率为0.5
    transform.RandomVerticalFlip(),
    # transform.ColorJitter(),
    transform.ToTensor()  # 转化数据类型为tensor
])
transforms_test = transform.Compose([
    #transform.Grayscale(),
    transform.Resize([320,320]),
    transform.ToTensor()
])
train_data = MyDataset(txt='C:/Users/cyw/Desktop/c20201105/train.txt', transform=transforms)
test_data = MyDataset(txt='C:/Users/cyw/Desktop/c20201105/test.txt', transform=transforms_test)
# 调用DataLoader和刚刚创建的数据集，来创建dataloader，loader的长度是有多少个batch，所以和batch_size有关
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=False)
print('num_of_trainData:', len(train_data))
print('num_of_testData:', len(test_data))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 120
lr = 0.000125
batch_size = 16




model = ResNet(Bottleneck,Bottleneck1, [3, 4, 6, 3]).to(device)
#print(model.state_dict().keys())
# 计算误差与
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4)


# 用于更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 训练
total_step = len(train_loader)
curr_lr = lr
bets_acc = 0.0
bets_epoch = 0
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将数据放入GPU
        model.train()
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1020 == 0:
            print("Epoch [{}/{}],step[{}/{}] Loss:{:.4f}"
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
    print('epoch{} loss{}'.format(epoch + 1, loss.item()))

    # 降低学习速度
    if (epoch + 1) % 40 == 0:  # 每过20个Epoch，学习率就会下降
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# 测试
    model.eval()
    with torch.no_grad():
        # correct = 0
        # total = 0
        # truth_class = 0
        num_correct = 0
        for i,(images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            num_correct += (predicted == labels).sum().item()
            if num_correct/len(test_data) > bets_acc:
                bets_acc = num_correct/len(test_data)
                bets_epoch = epoch+1
             #   torch.save(model,'nets-{}'.format(epoch))
        print('test_acc:{:4f}'.format(num_correct/len(test_data)))
print('best_test_acc:{}   best_epoch:{}'.format(bets_acc,bets_epoch))
