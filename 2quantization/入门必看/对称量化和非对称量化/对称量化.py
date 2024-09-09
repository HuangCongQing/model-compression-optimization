'''
Description: 对称量化（zero_point = 0）
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2024-09-09 18:22:40
LastEditTime: 2024-09-09 18:23:57
FilePath: /model-compression-optimization/2quantization/入门必看/对称量化和非对称量化/对称量化.py
'''
import numpy as np

def saturete(x):
    return np.clip(x, -127, 127)

def scale_cal(x):
    max_val = np.max(np.abs(x))
    return max_val / 127

def quant_float_data(x, scale):
    xq = saturete(np.round(x/scale))
    return xq

def dequant_data(xq, scale):
    x = (xq * scale).astype('float32')
    return x

if __name__ == "__main__":
    np.random.seed(1)
    data_float32 = np.random.randn(3).astype('float32')
    print(f"input = {data_float32}")

    scale = scale_cal(data_float32)
    print(f"scale = {scale}")

    data_int8 = quant_float_data(data_float32, scale)
    print(f"quant_result = {data_int8}")
    data_dequant_float = dequant_data(data_int8, scale)
    print(f"dequant_result = {data_dequant_float}")

    print(f"diff = {data_dequant_float - data_float32}")
'''  
input = [ 1.6243454 -0.6117564 -0.5281718]
scale = 0.012790121431425801
quant_result = [127. -48. -41.]
dequant_result = [ 1.6243454 -0.6139258 -0.524395 ]
diff = [ 0.         -0.00216943  0.00377679]
'''