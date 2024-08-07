import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_sla_tensor(file_path, save_dir, resolution=0.25):
    # 加载张量数据
    sla_tensor = torch.load(file_path)
    
    # 转换为 numpy 数组
    sla_array = sla_tensor.numpy()
    
    # 确定坐标范围
    min_lat, max_lat = -90, 90
    min_lon, max_lon = -180, 180
    
    # 计算新分辨率下的经纬度网格
    lat_bins = np.linspace(min_lat, max_lat, sla_array.shape[0])
    lon_bins = np.linspace(min_lon, max_lon, sla_array.shape[1])
    lon_grid, lat_grid = np.meshgrid(lon_bins, lat_bins)
    
    # 打印形状以便调试
    print(f"lon_grid shape: {lon_grid.shape}, lat_grid shape: {lat_grid.shape}, sla_array shape: {sla_array.shape}")
    
    # 可视化
    plt.figure(figsize=(14, 7))
    plt.pcolormesh(lon_grid, lat_grid, sla_array, shading='auto', cmap='jet')  # 使用 'jet' 色图
    plt.colorbar(label='Sea Surface Height Anomaly (m)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('SSHA Visualization')

    # 确定保存路径
    file_name = os.path.basename(file_path).replace('.pt', '.png')
    save_path = os.path.join(save_dir, file_name)
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

# 示例调用
tensor_file_path = '/data/challenge/processed_ssha/2019/dt_global_twosat_phy_l4_20190101_vDT2021.pt'
save_directory = '/home/challenge/code/visualizations/'
os.makedirs(save_directory, exist_ok=True)

visualize_sla_tensor(tensor_file_path, save_directory)
