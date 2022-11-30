import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from vgg import vgg
import numpy as np

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--percent', type=float, default=0.5, # 剪枝percent
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to raw trained model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model = vgg()
if args.cuda:
    model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print(model)
total = 0 # 每层特征图个数总和
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]  # 当前层BN的w权重shape

bn = torch.zeros(total) # 拿到每个gamma值，每层特征图(channel)都会对应一个gama， belta
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone() # 将每一层的每个权重绝对值保存下来
        index += size

y, i = torch.sort(bn) #从小到大排序
thre_index = int(total * args.percent) # 总量的70%
thre = y[thre_index] # 选择70%百分比位置的参数的卡点阈值0.2529

# 对应channel BN层gamma置0 && 得到新的cfg配置参数
pruned = 0
cfg = [] # 修改生成新的网络配置参数cfg，并保存在pth中，用于下次refine微调(小于thre置为0)===============================
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.clone() # 即需要开辟新的存储地址而不是引用，可以用 clone() 进行深拷贝
        mask = weight_copy.abs().gt(thre).float().cuda() # 阈值判断得到0/1列表  gt：当前参数是不是大于阈值thre
        pruned = pruned + mask.shape[0] - torch.sum(mask) # 单纯输出计数，剪枝了多少
        m.weight.data.mul_(mask) # BN层gamma置0 ，但是没有删除=====
        m.bias.data.mul_(mask) #  偏置置0
        cfg.append(int(torch.sum(mask))) # 修改配置参数
        cfg_mask.append(mask.clone()) # 每一层满足阈值的0/1 list的list（嵌套list）
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total # 0.7

print('Pre-processing Successful!')


# 部分置0后先测试下效果 simple test model after Pre-processing prune (simple set BN scales to zeros)
def test():
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data) # 调用修改后的模型
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

test() # Test set: Accuracy: 1000/10000 (10.0%) 精度只有10%，还需要【再训练】


# 执行剪枝 Make real prune
print(cfg)
newmodel = vgg(cfg=cfg) # 剪枝后的模型（cfg新的参数）================================
newmodel.cuda()

# 为剪枝后的newmodel模型【赋值权重】！！！
layer_id_in_cfg = 0
start_mask = torch.ones(3) # 初始输入channel(1,1,1)
end_mask = cfg_mask[layer_id_in_cfg] # 输出channel    cfg_mask：每一层满足阈值的0/1 list的list（嵌套list）
# num(16)*[Conv,BN,ReLU](3)=48    'M'(4)*[Maxpool](1)=4 nn.Linear*1=1
for [m0, m1] in zip(model.modules(), newmodel.modules()): # Conv,BN,ReLU final:ReLU
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) # 得到满足条件的下标列表  np.argwhere:返回条件非0的数组元组的索引
        m1.weight.data = m0.weight.data[idx1].clone() # 只选择赋值有1的下标
        m1.bias.data = m0.bias.data[idx1].clone()
        m1.running_mean = m0.running_mean[idx1].clone()
        m1.running_var = m0.running_var[idx1].clone()
        # 修改为下一层的输入输出channel（为啥在BN层修改为下一层的input_channel和out_channel???）
        layer_id_in_cfg += 1
        start_mask = end_mask.clone() # 将当前层的输出channel变为输入channel（新层的input_channel）比如3
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg] # 得到下一层的输出channel（新层的out_channel） 比如64（可能有删减变为53）

    elif isinstance(m0, nn.Conv2d):
        # 权重w的shape(out_channel, in_channel, k1, k2)
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy()))) # 比如3
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #  比如64（可能有删减变为53）
        print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0])) # In shape: 48 Out shape:64
        # 注意weight维度为(out_channel, in_channel, k1, k2) ,其中kernel_size=(5, 5)，两个卷积层连接，下一层的输入维度就等于当前层的c!!!
        w = m0.weight.data[:, idx0, :, :].clone() # oldmodel # 只选择赋值有1的下标   data(out_channel, in_channel, k1, k2) note:  https://www.yuque.com/huangzhongqing/lxph5a/kdwd5q#iY7XX
        w = w[idx1, :, :, :].clone() # 将所需权重赋值到剪枝后的模型
        m1.weight.data = w.clone()
        # m1.bias.data = m0.bias.data[idx1].clone()
    elif isinstance(m0, nn.Linear): # 最后1层 10分类
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        m1.weight.data = m0.weight.data[:, idx0].clone()

# 新模型保存下来，【再训练再微调】微调完成剪枝模型
torch.save(
            {'cfg': cfg,  # 剪枝后的新的卷积配置参数，用于新的微调时，重新配置参数
            'state_dict': newmodel.state_dict()
            }, 
            args.save) # args.save=--save pruned.pth.tar

print(newmodel)
model = newmodel
test()