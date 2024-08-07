import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio

def visualize_tensor_data_as_gif(tensor_file, output_image_path):
    """
    可视化 Tensor 文件中的数据并保存为 GIF 动图，每一帧代表一个深度。

    参数：
    tensor_file (str): Tensor 文件的路径。
    output_image_path (str): 输出 GIF 动图的路径。
    """
    # 加载 Tensor 文件
    tensor_data = torch.load(tensor_file)
    
    # 提取叶绿素浓度和深度数据
    chl_tensor = tensor_data['chl']
    depth_tensor = tensor_data['depth']
    

    # 假设 chl_tensor 是原始数据，形状为 [19, 681, 1440]
    original_shape = chl_tensor.shape
    desired_shape = (original_shape[0], 720, original_shape[2])
    fill_start = 40  # 从索引40开始填充原始数据

    # 创建一个新的张量，形状为 [19, 720, 1440]，使用np.nan初始化
    new_chl_tensor = np.full(desired_shape, np.nan)

    # 将原始数据转换为numpy数组（如果原始数据是Tensor）
    if isinstance(chl_tensor, torch.Tensor):
        chl_tensor = chl_tensor.numpy()

    # 将原始数据复制到新数组的指定位置
    new_chl_tensor[:, fill_start-1:fill_start -1  + original_shape[1], :] = chl_tensor
    # 将结果转换回Tensor
    new_chl_tensor = torch.tensor(new_chl_tensor, dtype=torch.float)

    # 将 Tensor 数据转换为 NumPy 数组
    chl_grid = new_chl_tensor.numpy()
    depth_values = depth_tensor.numpy()


    # 现在 new_chl_tensor 的形状为 [19, 720, 1440]，且已经用nan填充了-90到-80度的部分
    
    # 创建经纬度网格
    lat_bins = np.linspace(-90, 90, chl_grid.shape[1])
    lon_bins = np.linspace(-180, 180, chl_grid.shape[2])
    lon_grid, lat_grid = np.meshgrid(lon_bins, lat_bins)
    
    # 打印深度数据以检查是否均匀分布
    print(f"Depth values: {depth_values}")
    
    # 创建 GIF 动图
    frames = []
    for i in range(chl_grid.shape[0]):
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(lon_grid, lat_grid, chl_grid[i], cmap='jet', shading='auto')
        plt.colorbar(label='Chlorophyll Concentration (mg/m^3)')
        plt.title(f'Chlorophyll Concentration at Depth: {depth_values[i]:.2f}m')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # 保存当前帧
        frame_path = f'/tmp/frame_{i}.png'
        plt.savefig(frame_path)
        plt.close()
        
        # 读取保存的帧并添加到帧列表中
        frames.append(imageio.imread(frame_path))
    
    # 将所有帧保存为 GIF 动图
    imageio.mimsave(output_image_path, frames, duration=0.5)
    print(f"GIF saved to {output_image_path}")

# 示例调用
tensor_file = '/data/challenge/processed_chla/2020/cmems_mod_glo_bgc_0.25deg_20200601_chl_180.00W-179.75E_80.00S-90.00N_0.51-53.85m.pt'
output_image_path = 'visualizations/chla_20200601.gif'

# 可视化 Tensor 数据并保存为 GIF 动图
visualize_tensor_data_as_gif(tensor_file, output_image_path)
