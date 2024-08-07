import os
import numpy as np
import torch
from netCDF4 import Dataset, num2date
from tqdm import tqdm

def explore_nc_file(file_path):
    """
    查看 NetCDF 文件中的变量和维度信息。

    参数：
    file_path (str): NetCDF 文件的路径。
    """
    dataset = Dataset(file_path, 'r')
    
    print("Variables and dimensions in the NetCDF file:")
    for var in dataset.variables:
        print(f"Variable: {var}, Dimensions: {dataset.variables[var].dimensions}, Shape: {dataset.variables[var].shape}")
    
    dataset.close()


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
#    data = var[:] * scale_factor + offset
    data = var[:]
    data = np.ma.masked_equal(data, fill_value)  # 屏蔽填充值
    data = np.ma.filled(data, np.nan)  # 将屏蔽的值替换为 NaN
    
    dataset.close()
    return data


# 示例调用
input_file = '/data/challenge/sst/2019/20190101023149-NCEI-L3C_GHRSST-SSTskin-AVHRR_Pathfinder-PFV5.3_NOAA19_G_2019001_night-v02.0-fv01.0.nc'
explore_nc_file(input_file)


""" chl = read_and_convert_variable(input_file, 'chl')
# 检查原始叶绿素数据是否存在非 NaN 值
print(f"Original chlorophyll data min value: {np.nanmin(chl)}, max value: {np.nanmax(chl)}") """

