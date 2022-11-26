""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
from visualize import plot


config = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)

    net_crit = nn.CrossEntropyLoss().to(device)

    # 构建网络，即包括8个cell的supernet
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    model = model.to(device)

    # weights optimizer 用于网络参数 w 的优化器  
    # w_lr 初始值是0.025，使用的余弦退火调度更新学习率，每个epoch的学习率都不一样
    # w_momentum = 0.9 常用参数
    # w_weight_decay = 3e-4  正则化参数 
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)

    # alphas optimizer 用于结构参数 α 的优化器
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2   # 前一半data用于训练，后一半data用于验证
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)

    # 学习率更新参数，每次迭代调整不同的学习率   使用余弦退火调度设置各组参数组的学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)

    # 创建用于更新 α 的architect
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = 0.
    # 经历 epochs 次搜索后结束
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]   # 得到本次迭代的学习率lr

        model.print_alphas(logger)

        # training 先更新alpha，再更新w
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)

        # validation 前向传播看下训练的参数好坏
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step)

        # log
        # genotype
        genotype = model.genotype()   #对应论文2.4 选出来权重值大的两个前驱节点，并把(操作，前驱节点)存下来
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()     # 保存前 1 预测正确的概率
    top5 = utils.AverageMeter()     # 保存前 5 预测正确的概率
    losses = utils.AverageMeter()   # 保存loss值

    cur_step = epoch * len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    # 每个step取出一个batch，batchsize是64（256个数据对）
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        # 用于架构参数alpha 更新的一个batch, 使用iter(dataloader)返回的是一个迭代器, 然后可以使用next访问
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha) 对应伪代码的第 1 步, 结构参数梯度下降
        alpha_optim.zero_grad() # 清除之前学到的梯度的参数
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w) 对应伪代码的第 2 步, 网络参数梯度下降
        w_optim.zero_grad()     # 清除之前学到的梯度的参数
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)   # 预测值 logits 和真实值 target 的loss
        loss.backward()         # 反向传播，计算梯度

        # gradient clipping  梯度裁剪
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()      # 应用梯度

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


# 上一步train完model的参数已经更新了，我们就在验证集上前向传播一次求一下loss，看一下好坏
def validate(valid_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    main()
