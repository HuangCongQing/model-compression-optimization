'''
Description: 自动插入QDQ节点 https://www.yuque.com/huangzhongqing/lightweight/zt30g38wu35gk10y
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2024-09-09 15:51:59
LastEditTime: 2024-09-10 18:17:28
FilePath: /model-compression-optimization/2quantization/QAT/01[自动插入QDQ节点]resnet所有节点.py
'''
import torch
import torchvision
from pytorch_quantization import tensor_quant, quant_modules
from pytorch_quantization import nn as quant_nn

quant_modules.initialize()

model = torchvision.models.resnet18()
model.cuda()

inputs = torch.randn(1, 3, 224, 224, device='cuda')
#  将resnet18 模型中的所有节点替换为 QDQ 算子
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model, inputs, 'resnet18.onnx', opset_version=13)
