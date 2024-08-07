import os
import re
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
    
    # 读取比例因子和偏移量属性
    scale_factor = var.scale_factor if 'scale_factor' in var.ncattrs() else 1
    add_offset = var.add_offset if 'add_offset' in var.ncattrs() else 0
    
    # 读取填充值
    fill_value = var._FillValue if '_FillValue' in var.ncattrs() else np.nan
    
    # 读取数据
    data = var[:] * scale_factor + add_offset
    
    # 将填充值转换为 NaN
    data = np.where(data == fill_value, np.nan, data)
    
    dataset.close()
    return data

def process_single_day(file_day, file_night, output_file):
    """
    处理单个日期的SST数据（白天和夜晚），提取并保存为Tensor文件。

    参数：
    file_day (str): 白天数据的NetCDF文件路径。
    file_night (str): 夜晚数据的NetCDF文件路径。
    output_file (str): 输出Tensor文件的路径。
    """
    # 读取白天数据
    sst_day = read_and_convert_variable(file_day, 'sea_surface_temperature')
    
    # 读取夜晚数据
    sst_night = read_and_convert_variable(file_night, 'sea_surface_temperature')
    
    # 将数据转换为Tensor
    sst_day_tensor = torch.tensor(sst_day, dtype=torch.float32).squeeze()
    sst_night_tensor = torch.tensor(sst_night, dtype=torch.float32).squeeze()
    
    # 将温度范围限定在合理的范围内（例如 270 至 320 开尔文），并将无效值替换为 NaN
    valid_min = -180 * 0.01 + 273.15  # -180 摄氏度 转换为 开尔文
    valid_max = 4500 * 0.01 + 273.15  # 4500 摄氏度 转换为 开尔文
    sst_day_tensor = torch.where((sst_day_tensor < valid_min) | (sst_day_tensor > valid_max), torch.tensor(np.nan), sst_day_tensor)
    sst_night_tensor = torch.where((sst_night_tensor < valid_min) | (sst_night_tensor > valid_max), torch.tensor(np.nan), sst_night_tensor)
    
    # 创建三维Tensor，第三个维度表示时间段（0表示白天，1表示夜晚）
    sst_combined_tensor = torch.stack((sst_day_tensor, sst_night_tensor), dim=2)
    
    # 保存Tensor数据
    torch.save(sst_combined_tensor, output_file)

def process_year(year, input_dir, output_dir):
    """
    处理一整年的SST数据。

    参数：
    year (int): 年份。
    input_dir (str): 输入NetCDF文件的目录。
    output_dir (str): 输出Tensor文件的目录。
    """
    file_pattern = re.compile(r'(\d{8})(\d{6})-NCEI-L3C_GHRSST-SSTskin-AVHRR_Pathfinder-PFV5.3_NOAA19_G_\d{7}_(day|night)-v02.0-fv01.0.nc')
    files = os.listdir(input_dir)
    day_files = {}
    night_files = {}
    
    # 解析文件名，获取日期和时间段信息
    for file in files:
        match = file_pattern.match(file)
        if match:
            date = match.group(1)
            time_period = match.group(3)
            if time_period == 'day':
                day_files[date] = file
            elif time_period == 'night':
                night_files[date] = file
    
    # 处理每一天的数据
    for date in tqdm(day_files.keys(), desc=f"Processing {year}"):
        if date in night_files:
            file_day = os.path.join(input_dir, day_files[date])
            file_night = os.path.join(input_dir, night_files[date])
            output_file = os.path.join(output_dir, f"sst_{date}.pt")
            process_single_day(file_day, file_night, output_file)
        else:
            print(f"Missing night file for date {date}")

# 示例调用
input_dir_2019 = '/data/challenge/sst/2019/'
output_dir_2019 = '/data/challenge/processed_sst/2019/'
input_dir_2020 = '/data/challenge/sst/2020/'
output_dir_2020 = '/data/challenge/processed_sst/2020/'

os.makedirs(output_dir_2019, exist_ok=True)
os.makedirs(output_dir_2020, exist_ok=True)

# process_year(2019, input_dir_2019, output_dir_2019)
process_year(2020, input_dir_2020, output_dir_2020)
