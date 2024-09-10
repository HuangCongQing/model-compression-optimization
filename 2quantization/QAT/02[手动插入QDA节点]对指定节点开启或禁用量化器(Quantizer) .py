'''
Description: 【手动插入QDQ节点】 如何使能某些层插入 QDQ 节点，某些层不插入 QDQ 节点呢？在代码层面我们通过 disable_quantization 以及 enable_quantization 两个类来进行控制。
https://www.yuque.com/huangzhongqing/lightweight/zt30g38wu35gk10y
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2024-09-09 16:00:33
LastEditTime: 2024-09-10 18:17:46
FilePath: /model-compression-optimization/2quantization/QAT/02[手动插入QDA节点]对指定节点开启或禁用量化器(Quantizer) copy.py
'''
import torch
import torchvision
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from typing import List, Callable, Union, Dict

class disable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)    
    
    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    def __init__(self, model):
        self.model = model
    
    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled
    
    def __enter__(self):
        self.apply(True)
        return self
    
    def __exit__(self, *args, **kwargs):
        self.apply(False)

def quantizer_state(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print(name, module)

quant_modules.initialize()  # 对整个模型进行量化
model = torchvision.models.resnet50()
model.cuda()

# 我们对模型的 conv1 模块禁用量化器对该模块的量化，在导出的 ONNX 模型中可以看到该节点没有被插入 QDQ 节点量化
disable_quantization(model.conv1).apply() # 关闭某个节点的量化
# enable_quantization(model.conv1).apply() # 开启某个节点的量化
inputs = torch.randn(1, 3, 224, 224, device='cuda')
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model, inputs, 'quant_resnet50_disableconv1.onnx', opset_version=13)
