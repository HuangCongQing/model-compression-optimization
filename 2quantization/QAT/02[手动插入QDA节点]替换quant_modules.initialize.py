'''
Description: 之前我们提到对整个模型插入 QDQ 节点我们是通过 quant_modules.initialize() 来实现的，我们能否自定义实现整个模型的 QDQ 节点插入呢？而不用上述方法，官方提供的接口可控性、灵活度较差，我们自己来实现整个过程。
https://www.yuque.com/huangzhongqing/lightweight/zt30g38wu35gk10y
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2024-09-09 16:00:33
LastEditTime: 2024-09-10 18:17:35
FilePath: /model-compression-optimization/2quantization/QAT/02[手动插入QDA节点]替换quant_modules.initialize.py
'''
import torch
import torchvision
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from typing import List, Callable, Union, Dict

# transfer_torch_to_quantization 函数的作用是将原始模型的一个层转换成对应的量化层。
# 该函数首先创建一个新的量化层实例 quant_instance，然后将原始层的所有属性复制到这个实例中。
# 接着根据不同的 OP 算子类型来进行初始化，具体根据原始层是否有 weight，来初始化 quant_instance 的 input_quantizer 和 weight_quantizer 两个属性。
# 最后，将 quant_instance 返回。
def transfer_torch_to_quantization(nninstace : torch.nn.Module, quantmodule):
    
    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstace).items():
        setattr(quant_instance, k, val) # 继承所有的属性
    
    def __init__(self):

        if isinstance(self, quant_nn_utils.QuantInputMixin): # 只有input，没有weight
            quant_desc_input = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True)
            self.init_quantizer(quant_desc_input)

            # Turn on torch hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                # 开启 _torch_hist 可以使用 PyTorch 内置的直方图函数来提高校准速度
                self._input_quantizer._calibrator._torch_hist  = True  # 提速！！！！！！！！！
                self._weight_quantizer._calibrator._torch_hist = True  #
    
    __init__(quant_instance)
    return quant_instance
            
# replace_to_quantization_module 函数的作用是将原始模型中的指定层替换成对应的量化层，并返回替换后的模型。
# 具体来说，该函数遍历整个模型的层，如果当前层是被替换层，则调用 transfer_torch_to_quantization 函数将其转换为量化层。
def replace_to_quantization_module(model : torch.nn.Module, ignore_policy : Union[str, List[str], Callable] = None):
    
    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod
    
    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path      = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])
        
    recursive_and_replace_module(model)
            

# quant_modules.initialize() # 如何实现自定义QDQ节点插入？
model = torchvision.models.resnet50()
model.cuda()

replace_to_quantization_module(model)
inputs = torch.randn(1, 3, 224, 224, device='cuda')
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model, inputs, 'quant_resnet50_replace_to_quantization.onnx', opset_version=13)
