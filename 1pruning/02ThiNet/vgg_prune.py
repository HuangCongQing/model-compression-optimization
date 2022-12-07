import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn as nn
from load_dataset import load_dataset

from vgg_net import vgg
from utils import *
from trainer import *
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
# load pre-train model
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

print("-----Before Prune-----")
# print_model_parameters(model)
print_model_param_nums(model)
print_model_param_flops(model)
t1 = time.time()
test(model,test_loader,cuda=cuda)
t2 = time.time()
print("剪枝前测试所需时间：",(t2-t1)*1000)


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

# paper: ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression
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
conv_r = 0.3
linear_r = 0.7
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
mask = np.random.choice(x.shape[0],16)
x = x[mask]
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
        if linear_i ==0:
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
test(newmodel,test_loader=test_loader,cuda=cuda)
t2 = time.time()
print("剪枝后测试所需时间：",(t2-t1)*1000)

finetune_model = vgg(num_classes=10,depth=19,cfg=newmodel.config[0],fc=newmodel.config[1]).cuda()
finetune_model.load_state_dict(newmodel.state_dict())

optimizer = optim.Adam(finetune_model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

print("-----Fine Tune-----")
t1 = time.time()
best_prec1 = 0.
for epoch in range(40):
    train(epoch,finetune_model,optimizer,scheduler=scheduler,train_loader=train_loader,cuda=cuda,log_interval=log_interval)
    prec1 = test(finetune_model,test_loader=test_loader,cuda=cuda)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    if is_best:
        torch.save( finetune_model.state_dict(), os.path.join(save, 'model_cifar10_pruned.pth.tar'))
print(best_prec1)
t2 = time.time()
print("微调所花时间为：",int(t2-t1))


"""
dataset: cifar100   net: vgg19      lr: 0.01      batch_size: 64      test_batch_size: 256     微调epoch: 20       optimizer: SGD
                    剪枝前精度    剪枝后精度    微调后精度    剪枝前模型大小      剪枝后模型大小      剪枝前FLOPS     剪枝后FLOPS     样本大小
卷积层(0.5)           64.2%          0.9%        61%          150MB              95MB               1.24G          0.35G          16
卷积层(0.3)           64.5%          2.3%        64.5%        150MB              112MB              1.24G          0.63G          16
卷积层(0.3)+fc(0.5)   64.4%          1.6%        64.1%        150MB              60.1MB             1.24G          0.60G          16
卷积层(0.3)+fc(0.75)  64.4%          1.3%        63.9%        150MB              45.7MB             1.24G          0.59G          16
卷积层(0.5)+fc(0.85)  64.3%          1.0%        62.1%        150MB              24.2MB             1.24G          0.31G          16
卷积层(0.5)+fc(0.85)  64.2%          1.0%        62.4%        150MB              24.2MB             1.24G          0.31G          256
"""

"""
dataset: cifar10    net: vgg19  lr: 0.01  batch_size: 64  test_batch_size: 256 微调epoch: 20 optimizer: SGD 训练时长(epoch=20):1026s(17m42s)
                    剪枝前精度  剪枝后精度  微调后精度  剪枝前模型大小  剪枝后模型大小  剪枝前FLOPS 剪枝后FLOPS    剪枝时长  训练时长  微调时长  剪枝前测试时长  剪枝后测试时长  样本大小
卷积层(0.5)           92.1%      10%        88%         148MB           93.6MB      1.24G        0.35G         59s     1026s     574s       3490.2ms      2615.7ms       16
卷积层(0.3)           92.0%      19%        90.2%       148MB           111MB       1.24G        0.63G         29s     1026s     723s       3488.2ms      3087.2ms       16
卷积层(0.3)+fc(0.5)   92.1%      10.4%      90.27%      148MB           59.4MB      1.24G        0.60G         29s     1026s     650s       3475.4ms      3042.2ms       16
卷积层(0.3)+fc(0.75)  92.1%      10%        90.1%       148MB           45.3MB      1.24G        0.59G         29s     1026s     649s       3534.2ms      2954.1ms       16
卷积层(0.5)+fc(0.85)  92.1%      10%        88.7%       148MB           24.0MB      1.24G        0.31G         59s     1026s     570s       3458.3ms      2517.9ms       16
卷积层(0.5)+fc(0.85)  92.0%      10%        88.79%      148MB           24.0MB      1.24G        0.31G         647s    1026s     552s       3466.3ms      2463.6ms       256
"""

"""
dataset: cifar10    net: vgg19  lr: 0.01  batch_size: 64  test_batch_size: 256 微调epoch: 40 optimizer: SGD 训练时长(epoch=40):2064s(34m24s)
                    剪枝前精度  剪枝后精度  微调后精度  剪枝前模型大小  剪枝后模型大小  剪枝前FLOPS 剪枝后FLOPS    剪枝时长  训练时长  微调时长  剪枝前测试时长  剪枝后测试时长  样本大小
卷积层(0.3)+fc(0.7)   92.1%      10.1%      90.9%      148MB           47.5MB      1.24G        0.59G         269s     2064s     1282s       3457.7ms      2994.1ms       256
"""