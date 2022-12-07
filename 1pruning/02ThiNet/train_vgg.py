from __future__ import print_function

import os


import torch

import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from load_dataset import load_dataset

from vgg_net import vgg


# Training settings

dataset='cifar10' # 或者cifar100
cuda = True and torch.cuda.is_available()
batch_size = 64
test_batch_size = 256
save = './model_save'
lr = 0.01
momentum = 0.9
weight_decay = 1e-4
log_interval = 100
epochs = 200

torch.manual_seed(1)
if cuda:
    torch.cuda.manual_seed(1)

if not os.path.exists(save):
    os.makedirs(save)

train_loader, test_loader = load_dataset(dataset=dataset,cuda=cuda,batch_size=batch_size,test_batch_size=test_batch_size)


model = vgg(num_classes=10,depth=19)

if cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

tb = SummaryWriter("vgg_tb/")
init_img = torch.zeros((1, 3, 32, 32))
if cuda:
    init_img = init_img.cuda()
tb.add_graph(model, init_img)


def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
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
    
    tb.add_scalar("Loss/Train", avg_loss, epoch)
    tb.add_scalar("Correct/Train", train_acc, epoch)
    tb.add_scalar("Accuracy/Train", train_acc/ len(train_loader.dataset), epoch)
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        tb.add_histogram(name, tensor, epoch)

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        tb.add_scalar("Loss/Test", test_loss, epoch)
        tb.add_scalar("Correct/Test", correct, epoch)
        tb.add_scalar("Accuracy/Test", correct / float(len(test_loader.dataset)), epoch)

        return correct / float(len(test_loader.dataset))

best_prec1 = 0.
for epoch in range(epochs):
    train(epoch)
    prec1 = test(epoch)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    if is_best:
        torch.save({'config': model.config, 'state_dict': model.state_dict()}, os.path.join(save, 'model_cifar10.pth.tar'))

tb.close()