import os
import torch

def check_tensor_file(tensor_file):
    """
    检查并打印 Tensor 文件中的数据。

    参数：
    tensor_file (str): Tensor 文件的路径。
    """
    # 加载 Tensor 文件
    tensor_data = torch.load(tensor_file)
    
    # 提取数据
    chl_tensor = tensor_data['chl']
    depth_tensor = tensor_data['depth']
    
    # 将 Tensor 数据转换为 NumPy 数组
    chl_grid = chl_tensor.numpy()
    depth_values = depth_tensor.numpy()
    
    # 打印张量的形状
    print(f"Shape of Chlorophyll Tensor: {chl_grid.shape}")
    print(f"Shape of Depth Tensor: {depth_values.shape}")
    
    # 打印部分数据以进行检查
    print("\nChlorophyll Tensor (partial):")
    print(chl_grid[:, :5, :5])  # 打印部分数据
    print("\nDepth Values:")
    print(depth_values)

# 示例调用
tensor_file = '/data/challenge/processed_chla/2019/cmems_mod_glo_bgc_0.25deg_20190629_chl_180.00W-179.75E_80.00S-90.00N_0.51-53.85m.pt'

# 检查 Tensor 文件内容
# check_tensor_file(tensor_file)
tensor1 = torch.load(tensor_file)
print(tensor1)
print(tensor1.shape)