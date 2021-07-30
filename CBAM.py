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

def default_loader(path):
    return Image.open(path).convert('RGB')

def conv1x1(inplanes,outplaens,stride=1):
    return nn.Conv2d(inplanes,outplaens,kernel_size=1,stride=stride,bias=False)

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


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))

        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x

        out = self.spatial_attention(out) * out
        return out


class ResBlock_CBAM(nn.Module):
    expansion = 4
    def __init__(self,in_places, places, stride=1,downsampling=False, expansion = 4,norm_layer=None):
        super(ResBlock_CBAM,self).__init__()
        #self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )
        self.cbam = CBAM(channel=places*self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = ResNet(ResBlock_CBAM,ResBlock_CBAM,[3,4,6,3])
net.to(device)
print(net)

transforms = transform.Compose([
    # transform.Grayscale(),
    transform.Resize([320,320]),
    transform.RandomHorizontalFlip(),  # 随机水平翻转，概率为0.5
    transform.RandomVerticalFlip(),
    #transform.ColorJitter(),
    transform.ToTensor()  # 转化数据类型为tensor
])
transforms_test = transform.Compose([
    # transform.Grayscale(),
    transform.Resize([320,320]),
    transform.ToTensor()
])
train_data = MyDataset(txt='C:/Users/cyw/Desktop/c20201105/train.txt', transform=transforms)
test_data = MyDataset(txt='C:/Users/cyw/Desktop/c20201105/test.txt', transform=transforms_test)
# 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=False)
print('num_of_trainData:', len(train_data))
print('num_of_testData:', len(test_data))


epochs = 120
lr = 0.00025
batch_size = 2


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr,momentum=0.95,weight_decay=5e-4)


# 用于更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_step = len(train_loader)
curr_lr = lr
bets_acc = 0.0
bets_epoch = 0
test_acc = []
train_acc = []
name=[]
milestones = [40,60,80,100]
#wrong = open('C:/Users/cyw/Desktop/wrong2.txt', 'r+')
classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8')
# 训练
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

    # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.2, last_epoch=-1)

    # 降低学习速度
    if (epoch + 1) % 40 == 0 :  # 每过20个Epoch，学习率就会下降
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
    class_correct = list(0. for i in range(12))
    class_total = list(0. for i in range(12))
# 测试
    name.clear()

    with torch.no_grad():
        net.eval()
        num_correct = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            num_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()

            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                if predicted[i]!=labels[i]:
                    name.append((str(int(predicted[i])),str(int(labels[i]))))

        if num_correct / len(test_data) > bets_acc:
            #wrong.seek(0)
            #wrong.truncate(0)
            bets_acc = num_correct / len(test_data)
            bets_epoch = epoch + 1
            classc = copy.deepcopy(class_correct)
            classt = copy.deepcopy(class_total)
            # 计入错判样本
            for x1,x2 in name:
                x = str(x1)+str(x2)+'\n'
                #wrong.write(x)
           # torch.save(net,'net-{}'.format(epoch))
    print('acc:{:4f} {}/{}'.format(num_correct/len(test_data),num_correct,len(test_data)))
    test_acc.append(num_correct/len(test_data))
print('best_acc:{}   best_epoch:{}'.format(bets_acc,bets_epoch))
# 输出各类准确度
try:
    for i in range(9):
        print('Accuracy of %5s : %2d %% %2d/%2d ' % (
            classes[i], 100 * classc[i] / classt[i],classc[i], classt[i]))
except NameError:
    for i in range(9):
        print('Accuracy of %5s : %2d %% %2d/%2d ' % (
            classes[i], 100 * class_correct[i] / class_total[i],class_correct[i], class_total[i]))
#wrong.close()
# 准确度可视化
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
