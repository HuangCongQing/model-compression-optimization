""" CNN cell for architecture search 
    定义了搜索空间, darts要搜索cell的结构, 用来构建CNN 
    
    第k个cell结构: 
    2 个 input node: 连接前两个cell的output node
    4 个 intermediate node: 与所有前驱node相连
    1 个 output node: 对 4 个intermediate node进行 concat
    
    初始化cell流程: 
    1. 判断第 k-1 个cell是否reduction, 是则preproc0 需要缩减尺寸, 否则preproc0 为标准conv。 preproc1 对第 k-1 个cell的输出做预处理, 为标准conv
    2. dag中保存 4个 intermediate node的输入边, 分别有 2、3、4、5条边, 每条边是8种操作的混合计算结果
    
    前向传播: 
    1. 给定第 k-2 和 k-1 个cell的输出, 为s0, s1, 分别经过preproc0, preproc1预处理, 保存在states中
    2. 计算intermediate node 0 的值, 即 s0, s1 经过mixedOp操作结果 加权求和, 记作 n0 加入 states
    3. 计算intermediate node 1 的值, 即 s0, s1, n0 经过mixedOp操作结果 加权求和,  记作 n1 加入 states
    4. 计算intermediate node 2 的值, 即 s0, s1, n0, n1 经过mixedOp操作结果 加权求和, 记作 n2 加入 states
    5. 计算intermediate node 3 的值, 即 s0, s1, n0, n1, n2 经过mixedOp操作结果 加权求和, 记作 n3 加入 states
    6. 对 n0~n3 的 output 进行concat作为当前cell的输出, 通道数从 C 变成 4C
"""
import torch
import torch.nn as nn
from models import ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction):
        """
        Args:  
            n_nodes: # of intermediate n_nodes  中间节点个数
            C_pp: C_out[k-2]    第 k-2 个cell的输出通道数, 与输入node 1 相连
            C_p : C_out[k-1]    第 k-1 个cell的输出通道数, 与输入node 2 相连
            C   : C_in[k] (current)     当前是第 k 个cell, 输入通道数为 C
            reduction_p: flag for whether the previous cell is reduction cell or not  前一个cell是否缩小了尺寸
            reduction: flag for whether the current cell is reduction cell or not     当前cell是否缩小了尺寸
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes   # 默认=4, 每个cell中有4个中间节点的连接状态待确定

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing. 
        # 决定第 1 个input node的结构
        if reduction_p:  
            # 如果第 k-1 个cell是reduction cell, 前面的输出尺寸是缩小的, 因此k-2和k的尺寸也要缩小
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        else: 
            # 直接使用第 k -2 个的通道数 作为操作的输入通道数, 是cell k-2 的输出
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        # 决定第 2 个input node的结构, 是cell k-1 的输出
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag 
        # 构建operation的modulelist, 包括4个列表, 每个列表长度依次为 2, 3, 4, 5, dag的总长度为 14
        self.dag = nn.ModuleList()  

        # 遍历 4 个中间节点node, i = 0, 1, 2, 3
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            # 对第 i 个node来说, 他有 j 个前驱node  
            # 每个node的input都由 前 2 个cell的输出 和 当前cell的前面的node组成 (0..i-1)
            # 例如 i = 1 时,  j = 0, 1, 2, 其中 2 是前面的node i = 0
            for j in range(2 + i):  
                # 只有自身为 reduction cell, 且处理前 2 个cell的输出时, 才扩大步长, 缩小尺寸
                stride = 2 if reduction and j < 2 else 1
                # op 是构建两个节点之间的混合操作, 即 8 种操作的结果求和
                op = ops.MixedOp(C, stride)
                # 所有8 条边的混合操作, 添加到这个节点i的列表中  
                self.dag[i].append(op)
                # 例如 dag[1] 的长度为 3, 表示有3条边与node 1相连

    # 前向传播时自动调用, cell中的计算过程
    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        # 第 k-2, k-1 个cell的输出经过操作后成为 2个 input node
        states = [s0, s1] 

        # 已知 2 个 input node的结果, 遍历每条边 edges, 共 14 条, 得到每个中间 node i 的 output
        for edges, w_list in zip(self.dag, w_dag):  
            # e.g node i = 0, s = (s0, s1) 分别求 node 0 对应的 2 条边的计算结果（调用了MixedOp的forward）, 再把结果相加
            # e.g node i = 1, s = (s0, s1, s2) 不仅求与input node相连的 2 条边的结果, 还求与 i = 0 相连的边的结果
            # s 理解成 node 的 output 或者 edge 的 input
            # w 是这条边的权重, 经过计算后得到边尾部 node 的输出, 即下一条边的 input
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))

            #把node i 的output 作为下一个node的输入
            states.append(s_cur)
            # states 包括 input node 0, 1, intermediate node 0, 1, 2, 3 的输出结果

        # 对intermediate node 的 output 进行concat作为当前cell的输出
        # dim = 1 是指对通道这个维度concat, 所以输出的通道数变成原来的 4 倍
        s_out = torch.cat(states[2:], dim = 1)
        return s_out
