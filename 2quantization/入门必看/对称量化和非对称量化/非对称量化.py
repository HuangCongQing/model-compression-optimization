'''
Description:  正常模型量化的流程是计算Scale&Zero_point=>量化=>截断=>反量化 https://blog.csdn.net/qq_40672115/article/details/129812691
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2024-09-09 18:09:12
LastEditTime: 2024-09-09 18:23:17
FilePath: /model-compression-optimization/2quantization/入门必看/对称量化和非对称量化/非对称量化.py
'''
import numpy as np

# 3 截断
def saturete(x, int_max, int_min):
    return np.clip(x, int_min, int_max)

# 1 Scale&Zero_point
def scale_z_cal(x, int_max, int_min):
    scale = (x.max() - x.min()) / (int_max - int_min)
    z = int_max - np.round((x.max() / scale))
    return scale, z

# 2 量化
def quant_float_data(x, scale, z, int_max, int_min):
    xq = saturete(np.round(x/scale + z), int_max, int_min)
    return xq

# 4 反量化
def dequant_data(xq, scale, z):
    x = ((xq - z)*scale).astype('float32')
    return x

if __name__ == "__main__":
    np.random.seed(1)
    data_float32 = np.random.randn(3).astype('float32')
    int_max = 127
    int_min = -128
    print(f"input = {data_float32}")

    scale, z = scale_z_cal(data_float32, int_max, int_min)
    print(f"scale = {scale}")
    print(f"z = {z}")
    data_int8 = quant_float_data(data_float32, scale, z, int_max, int_min)
    print(f"quant_result = {data_int8}")
    data_dequant_float = dequant_data(data_int8, scale, z)
    print(f"dequant_result = {data_dequant_float}")
    
    print(f"diff = {data_dequant_float - data_float32}")

'''  
input = [ 1.6243454 -0.6117564 -0.5281718]
scale = 0.008769026924582089
z = -58.0
quant_result = [ 127. -128. -118.]
dequant_result = [ 1.62227    -0.6138319  -0.52614164]
diff = [-0.00207543 -0.00207549  0.00203013]
'''