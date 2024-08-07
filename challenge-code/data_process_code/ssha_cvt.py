import os
import numpy as np
import torch
from netCDF4 import Dataset
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
    # data = (var[:]-offset)/ scale_factor 
    data = var[:]
    data = np.ma.masked_equal(data, fill_value)  # 屏蔽填充值
    
    dataset.close()
    return data

def process_single_ssha_file(file_path, output_file):
    """
    处理单个NetCDF文件，提取经纬度和SSHA数据并保存为Tensor文件。

    参数：
    file_path (str): 输入NetCDF文件的路径。
    output_file (str): 输出Tensor文件的路径。
    """
    dataset = Dataset(file_path, 'r')
    
    # 读取变量
    latitude = dataset.variables['latitude'][:]
    longitude = dataset.variables['longitude'][:]
    sla = read_and_convert_variable(file_path, 'sla')
    
    # 创建网格数据
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    
    # 将数据转换为Tensor
    sla_tensor = torch.tensor(sla[0].filled(np.nan), dtype=torch.float32)  # 去掉第一个维度
    
    # 保存Tensor数据
    torch.save(sla_tensor, output_file)
    print(f"Tensor data saved to {output_file}")

def process_all_files(input_dir, output_dir):
    """
    批量处理所有NetCDF文件，提取数据并转换为Tensor格式。

    参数：
    input_dir (str): 输入NetCDF文件的目录。
    output_dir (str): 输出Tensor文件的目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nc')])
    
    for file in tqdm(files, desc="Processing files"):
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file.replace('.nc', '.pt'))
        process_single_ssha_file(input_file, output_file)

# 示例调用
input_dir_2019 = '/data/challenge_data/原始数据/ssha/2019'
output_dir_2019 = '/data/challenge_data/处理数据/processed_ssha/2019/'

input_dir_2020 = '/data/challenge_data/原始数据/ssha/2020'
output_dir_2020 = '/data/challenge_data/处理数据/processed_ssha/2020/'

# 处理2019年的所有文件
process_all_files(input_dir_2019, output_dir_2019)

# 处理2020年的所有文件
process_all_files(input_dir_2020, output_dir_2020)
