import torch
import torch.nn as nn
from torch.autograd import Variable
import math  # init

# 总层数：48+4+1=53
# ●  num(16)*[Conv,BN,ReLU](3)=48
# ● 'M'(4)*[Maxpool](1)=4
# ● nn.Linear*1=1
class vgg(nn.Module):

    def __init__(self, dataset='cifar10', init_weights=True, cfg=None):
        super(vgg, self).__init__()
        # 配置参数
        if cfg is None:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
        # num(16)*[Conv,BN,ReLU](3)=48 'M'(4)*[Maxpool](1)=4 nn.Linear*1=1
        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'cifar10':
            num_classes = 10
        self.classifier = nn.Linear(cfg[-1], num_classes) # 最后 10分类
        if init_weights:
            self._initialize_weights()

    # 构建神经网络层
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False) # 2d卷积
                # BN层
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)] # Conv,BN,ReLU
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x) # make_layers
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x) # 10分类
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5) # BN层w初始化参数
                m.bias.data.zero_() # 偏置b
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01) # 线性层参数
                m.bias.data.zero_()


if __name__ == '__main__':
    net = vgg()
    x = Variable(torch.FloatTensor(16, 3, 40, 40))
    y = net(x)
    print(y.data.shape)

