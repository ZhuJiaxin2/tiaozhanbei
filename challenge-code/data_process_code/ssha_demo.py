import os
import numpy as np
import torch
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_and_convert_variable(nc_file, var_name):
    """
    读取并转换 NetCDF 文件中的变量，考虑比例因子和偏移量。
    
    参数：
    nc_file (str): NetCDF 文件的路径。
    var_name (str): 要读取的变量名。
    
    返回：
    data (numpy.ndarray): 转换后的数据。
    """
    dataset = Dataset(nc_file, 'r')
    var = dataset.variables[var_name]
    
    # 读取比例因子和偏移量属性，如果存在的话
    scale_factor = var.scale_factor if 'scale_factor' in var.ncattrs() else 1
    offset = var.add_offset if 'add_offset' in var.ncattrs() else 0
    
    # 读取填充值
    fill_value = var._FillValue if '_FillValue' in var.ncattrs() else np.nan
    
    # 读取数据并进行转换
    data = var[:] * scale_factor + offset
    data = np.ma.masked_equal(data, fill_value)  # 屏蔽填充值
    
    dataset.close()
    return data

def process_single_ssha_file(file_path, output_file, spatial_resolution=0.25):
    """
    处理单个NetCDF文件，提取经纬度和SSHA数据并保存为Tensor文件。

    参数：
    file_path (str): 输入NetCDF文件的路径。
    output_file (str): 输出Tensor文件的路径。
    spatial_resolution (float): 空间分辨率，默认值为0.25度。
    """
    dataset = Dataset(file_path, 'r')
    
    # 读取变量
    latitude = dataset.variables['latitude'][:]
    longitude = dataset.variables['longitude'][:]
    sla = read_and_convert_variable(file_path, 'sla')
    
    # 打印调试信息
    print(f"Shape of latitude: {latitude.shape}")
    print(f"Shape of longitude: {longitude.shape}")
    print(f"Shape of sla: {sla.shape}")
    print(f"SLA min value: {np.min(sla)}, max value: {np.max(sla)}")
    
    # 关闭NetCDF文件
    dataset.close()
    
    # 创建网格数据
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    
    # 将数据转换为Tensor
    lat_tensor = torch.tensor(lat_grid, dtype=torch.float32)
    lon_tensor = torch.tensor(lon_grid, dtype=torch.float32)
    sla_tensor = torch.tensor(sla[0].filled(np.nan), dtype=torch.float32)  # 去掉第一个维度
    
    # 保存Tensor数据
    tensor_data = {
        'latitude': lat_tensor,
        'longitude': lon_tensor,
        'sla': sla_tensor
    }
    torch.save(tensor_data, output_file)
    print(f"Tensor data saved to {output_file}")

def visualize_tensor_data(tensor_file, output_image_path):
    """
    可视化Tensor文件中的数据并保存图片。

    参数：
    tensor_file (str): Tensor文件的路径。
    output_image_path (str): 输出图片的路径。
    """
    tensor_data = torch.load(tensor_file)
    lat_tensor = tensor_data['latitude']
    lon_tensor = tensor_data['longitude']
    sla_tensor = tensor_data['sla']

    # 打印调试信息
    print(f"Shape of lat_tensor: {lat_tensor.shape}")
    print(f"Shape of lon_tensor: {lon_tensor.shape}")
    print(f"Shape of sla_tensor: {sla_tensor.shape}")
    print(f"SLA Tensor min value: {torch.min(sla_tensor)}, max value: {torch.max(sla_tensor)}")

    # 将Tensor数据转换为NumPy数组
    lat_grid = lat_tensor.numpy()
    lon_grid = lon_tensor.numpy()
    sla_grid = sla_tensor.numpy()

    # 创建图形并保存到指定路径
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(lon_grid, lat_grid, sla_grid, cmap='jet', shading='auto')
    plt.colorbar(label='Sea Surface Height Anomaly (m)')
    plt.title('SSHA Visualization')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(output_image_path)
    plt.close()
    print(f"Visualization saved to {output_image_path}")

# 示例调用
input_file = '/data/challenge/ssha/2019/dt_global_twosat_phy_l4_20190101_vDT2021.nc'
output_file = '/data/challenge/processed_ssha/2019/dt_global_twosat_phy_l4_20190101_vDT2021.pt'
output_image_path = '/home/challenge/code/visualizations/ssha_20190101.png'

# 处理单个文件
process_single_ssha_file(input_file, output_file)

# 可视化处理后的Tensor数据并保存图片
visualize_tensor_data(output_file, output_image_path)
