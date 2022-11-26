""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
import genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels   输入通道数
            C: # of starting model channels 初始通道数
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell    默认为4个中间节点
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C          
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C     # 当前Sequential模块的输出通道数
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),    # 前三个参数分别是输入图片的通道数，卷积核的数量，卷积核的大小
            nn.BatchNorm2d(C_cur)        # BatchNorm2d 对 minibatch 3d 数据组成的 4d 输入进行batchnormalization操作，num_features为(N,C,H,W)的C
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False     # 连接的前一个cell 不是 reduction cell
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True    # 网络是8层，在1/3和2/3位置是reduction cell 其他是normal cell，reduction cell的stride是 2
            else:
                reduction = False

            # 构建cell  每个cell的input nodes是前前cell和前一个cell的输出
            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)    # 4NAS\01DARTS(ICLR2019)\pt.darts\models\search_cells.py
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out      # C_p = 4 * C_cur 是因为每个cell的输出是4个中间节点concat的，这个concat是在通道这个维度，所以输出的通道数变为原来的4倍

        self.gap = nn.AdaptiveAvgPool2d(1)          # 构建一个平均池化层，output size是1x1
        self.linear = nn.Linear(C_p, n_classes)     # 构建一个线性分类器

        '''
        cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction) 
        layers = 8, 第2和5个cell是reduction_cell
        cells[0]: cell = Cell(4, 48,  48,  16, false,  false) 输出[N,16*4,h,w]
        cells[1]: cell = Cell(4, 48,  64,  16, false,  false) 输出[N,16*4,h,w]
        cells[2]: cell = Cell(4, 64,  64,  32, false,  True)  输出[N,32*4,h,w]
        cells[3]: cell = Cell(4, 64,  128, 32, false,  false) 输出[N,32*4,h,w]
        cells[4]: cell = Cell(4, 128, 128, 32, false,  false) 输出[N,32*4,h,w]
        cells[5]: cell = Cell(4, 128, 128, 64, false,  True)  输出[N,64*4,h,w]
        cells[6]: cell = Cell(4, 128, 256, 64, false,  false) 输出[N,64*4,h,w]
        cells[7]: cell = Cell(4, 256, 256, 64, false,  false) 输出[N,64*4,h,w]
        ''' 

    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, C_in, C, n_classes, n_layers, criterion, n_nodes=4, stem_multiplier=3,
                 device_ids=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        if len(self.device_ids) == 1:
            return self.net(x, weights_normal, weights_reduce)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies, wreduce_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
