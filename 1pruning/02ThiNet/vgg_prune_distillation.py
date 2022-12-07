import argparse
import numpy as np
import os
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from load_dataset import load_dataset

from vgg_net import vgg
from utils import *
import time

from compute_flops import print_model_param_flops,print_model_param_nums

# Prune settings
cuda = True and torch.cuda.is_available()
save = 'pruned_model'
dataset = 'cifar10'
batch_size = 64
test_batch_size = 256
log_interval = 100


if not os.path.exists(save):
    os.makedirs(save)

model_path = 'model_save/model_cifar10.pth.tar'
if model_path:
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        config = checkpoint['config']
        model = vgg(num_classes=10,cfg = config[0],fc=config[1])
        if cuda:
            model.cuda()
        model.load_state_dict(checkpoint['state_dict'])

print('Pre-processing Successful!')

train_loader, test_loader = load_dataset(dataset=dataset,cuda=cuda,batch_size=batch_size,test_batch_size=test_batch_size)



def test(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        return correct / float(len(test_loader.dataset))

# Knowledge_Distillation
def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

def train(epoch,model,teacher_model,optimizer,scheduler):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        teacher_output = teacher_model(data)
        teacher_output = teacher_output.detach() # cut off the backward of teacher model
        loss = distillation(output, target, teacher_output, temp=5.0, alpha=0.7)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                
    scheduler.step(avg_loss)
print("-----Before Prune-----")
# print_model_parameters(model)
print_model_param_nums(model)
print_model_param_flops(model)
t1 = time.time()
test(model)
t2 = time.time()
print("剪枝前一次测试所需时间：",(t2-t1)*1000)


def generate_windows(layer,last_output):
    output_h = last_output.shape[2]
    output_w = last_output.shape[3]
    windows_size_h = layer.kernel_size[0]
    windows_size_w = layer.kernel_size[1]
    H = output_h -windows_size_h
    W = output_w -windows_size_w
    if H < 0: H = 0
    if W < 0: W = 0
    x1 = np.random.randint(0,H + 1)
    x2 = x1 + windows_size_h
    y1 = np.random.randint(0,W + 1)
    y2 = y1 + windows_size_w
    s_w = last_output[:,:,x1:x2,y1:y2]
    return s_w

def get_T(s_w,w,r):
    
    T = []
    C = s_w.shape[1]
    I = [i for i in range(C)]
    pad = 0
    if s_w.shape[2] < w.shape[2]:
        pad = w.shape[2] - s_w.shape[2]
    s_w = np.pad(s_w.detach().numpy(),((0,0),(0,0),(0,pad),(0,pad)))
    w = w.detach().numpy()

    while len(T) < C * r:
        min_value = float("inf")
        for i in I:
            temT = T + [i]
            value = np.sum(np.sum((s_w[:,temT,:,:]*w[temT,:,:]),axis=(1,2,3))) ** 2
            if value < min_value:
                min_value = value
                min_i = i
        T.append(min_i)
        I.remove(i)
    return I


conv_i = 0
linear_i = 0
conv_r = 0.5
linear_r = 0.85
new_cfg = []
new_fc = []
for v in model.config[0]:
    a = v
    if a != "M":
        a = int(np.floor(v * (1-conv_r)))
    new_cfg.append(a)
for v in model.config[1]:
    a = v
    a = int(np.floor(v * (1-linear_r)))
    new_fc.append(a)
new_cfg[-1] = model.config[0][-1]

newmodel = vgg(num_classes=10,depth=19,cfg=new_cfg,fc=new_fc).cuda()

x = next(iter(test_loader))[0].cuda()
# mask = np.random.choice(x.shape[0],16)
# x = x[mask]

t1 = time.time()
for (m1,m2) in zip(model.modules(),newmodel.modules()):
    if isinstance(m1,nn.Conv2d): 
        if conv_i == 0:
            last_output = m1(x)
            last_conv_2 = m2
            last_conv_1 = m1
            last_T = [i for i in range(m1.weight.data.shape[1])]
            conv_i += 1
            continue
            
        print('Pruning the {0}th Conv layer'.format(conv_i))
        s_w = generate_windows(m1,last_output)
        random_filter = np.random.randint(0,m1.weight.data.shape[0])
        w = m1.weight.data[random_filter]
        T = get_T(s_w.cpu(),w.cpu(),conv_r)

        last_conv_2.weight.data = last_conv_1.weight.data[T].clone()
        last_conv_2.weight.data = last_conv_2.weight.data[:,last_T,:,:].clone()
        
        last_bn_2.weight.data = last_bn_1.weight.data[T].clone()
        last_bn_2.bias.data = last_bn_1.bias.data[T].clone()
        last_bn_2.running_mean = last_bn_1.running_mean[T].clone()
        last_bn_2.running_var = last_bn_1.running_var[T].clone()
        
        m2.weight.data = m1.weight.data[:,T,:,:].clone()

        last_output = last_output[:,T,:,:]
        last_output = m2(last_output)
        last_conv_2 = m2
        last_conv_1 = m1
        last_T = T
        conv_i += 1

    elif isinstance(m1,nn.BatchNorm2d):
        last_output = m1(last_output)
        last_bn_2 = m2
        last_bn_1 = m1
        
    elif isinstance(m1,nn.ReLU) or isinstance(m1,nn.MaxPool2d) or isinstance(m1,nn.Dropout):
        last_output = m1(last_output)
    elif isinstance(m1,nn.Linear):
        if linear_i == 0:
            last_linear_2 = m2
            last_linear_1 = m1
            linear_i += 1
            last_T = [i for i in range(m1.weight.data.shape[1])]
            continue
            
        print('Pruning the {0}th Linear layer'.format(linear_i))
        weight_copy = m1.weight.data.abs().clone()
        weight_copy = weight_copy.cpu().numpy()
        L1_norm = np.sum(weight_copy, axis=0)
        arg_max = np.argsort(L1_norm)
        alive_param_num = int(weight_copy.shape[1]*(1-linear_r))
        arg_max_rev = arg_max[::-1][:alive_param_num]

        last_linear_2.weight.data = last_linear_1.weight.data[arg_max_rev.tolist()].clone()
        last_linear_2.weight.data = last_linear_2.weight.data[:,last_T].clone()
        last_linear_2.bias.data = last_linear_1.bias.data[arg_max_rev.tolist()].clone()
        
        m2.weight.data = m1.weight.data[:,arg_max_rev.tolist()].clone()

        last_linear_2 = m2
        last_linear_1 = m1
        linear_i += 1
        last_T = arg_max_rev.tolist()
t2 = time.time()
print("剪枝所花时间：",int(t2-t1))
print("-----After Prune-----")
# print_model_parameters(newmodel)
print_model_param_nums(newmodel)
print_model_param_flops(newmodel)
t1 = time.time()
test(newmodel)
t2 = time.time()
print("剪枝后一次测试所需时间：",(t2-t1)*1000)


finetune_model = vgg(num_classes=10,depth=19,cfg=newmodel.config[0],fc=newmodel.config[1]).cuda()

finetune_model.load_state_dict(newmodel.state_dict())
optimizer = optim.SGD(finetune_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
print("-----Fine Tune-----")
t1 = time.time()
best_prec1 = 0.
for epoch in range(20):
    train(epoch,finetune_model,model,optimizer,scheduler)
    prec1 = test(finetune_model)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    if is_best:
        torch.save( finetune_model.state_dict(), os.path.join(save, 'model_cifar10_pruned.pth.tar'))
print(best_prec1)
t2 = time.time()
print("微调所花时间为：",int(t2-t1))


"""
dataset: cifar10    net: vgg19  lr: 0.01  batch_size: 64  test_batch_size: 256 微调epoch: 20 optimizer: SGD 训练时长(epoch=20):1026s(17m42s)
                    剪枝前精度  剪枝后精度  微调后精度  剪枝前模型大小  剪枝后模型大小  剪枝前FLOPS 剪枝后FLOPS    剪枝时长  训练时长  微调时长  剪枝前测试时长  剪枝后测试时长  样本大小
卷积层(0.5)+fc(0.85)  92.0%      10%        89.04%      148MB           24.0MB      1.24G        0.31G         647s    1026s     775ss       3466.3ms      2463.6ms      16
"""