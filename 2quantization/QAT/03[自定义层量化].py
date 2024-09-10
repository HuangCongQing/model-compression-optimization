'''
Description: 自定义层手动插入 QDQ 节点
自定义层分为两种，一种是只有 input 另一种是包含 input 和 weight
https://www.yuque.com/huangzhongqing/lightweight/zt30g38wu35gk10y
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2024-09-10 18:13:13
LastEditTime: 2024-09-10 18:17:39
FilePath: /model-compression-optimization/2quantization/QAT/03[自定义层量化].py
'''

import torch
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor

class QuantMultiAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histgoram"))
    
    def forward(self, x, y, z):
        return self._input_quantizer(x) + self._input_quantizer(y) + self._input_quantizer(z)

model = QuantMultiAdd()
model.cuda()
input_a = torch.randn(1, 3, 224, 224, device='cuda')
input_b = torch.randn(1, 3, 224, 224, device='cuda')
input_c = torch.randn(1, 3, 224, 224, device='cuda')
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model, (input_a, input_b, input_c), 'quantMultiAdd.onnx', opset_version=13)
