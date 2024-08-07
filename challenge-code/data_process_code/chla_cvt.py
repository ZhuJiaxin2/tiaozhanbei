import os
import numpy as np
import torch
from netCDF4 import Dataset, num2date

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
    # data = (var[:]-offset) / scale_factor 
    data = var[:]
    
    # 将填充值转换为 NaN
    data = np.where(data == fill_value, np.nan, data)
    
    dataset.close()
    return data

def process_all_days(file_path, output_dir):
    """
    处理一个NetCDF文件中的所有天的数据，提取经纬度和叶绿素浓度数据并保存为Tensor文件。

    参数：
    file_path (str): 输入NetCDF文件的路径。
    output_dir (str): 输出Tensor文件的目录。
    """
    dataset = Dataset(file_path, 'r')
    
    # 读取变量
    latitude = dataset.variables['latitude'][:]
    longitude = dataset.variables['longitude'][:]
    depth = dataset.variables['depth'][:]
    time = dataset.variables['time']
    
    # 获取时间戳对应的日期
    dates = num2date(time[:], time.units)
    
    # 读取叶绿素浓度数据
    chl = read_and_convert_variable(file_path, 'chl')
    
    # 创建网格数据
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    
    for day_index, date in enumerate(dates):
        date_str = date.strftime("%Y%m%d")
        
        # 提取当前日期的叶绿素浓度数据
        chl_day = chl[day_index]
        
        # 将填充值替换为 NaN
        chl_day = np.where(chl_day == 9.969209968386869e+36, np.nan, chl_day)
        
        # 将数据转换为Tensor
        chl_tensor = torch.tensor(chl_day, dtype=torch.float32)
        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        
        # 输出文件路径
        output_file = os.path.join(output_dir, f"cmems_mod_glo_bgc_0.25deg_{date_str}_chl_180.00W-179.75E_80.00S-90.00N_0.51-53.85m.pt")
        
        # 保存Tensor数据
        tensor_data = {
            'chl': chl_tensor,
            'depth': depth_tensor
        }
        torch.save(tensor_data, output_file)
        
    dataset.close()

# 示例调用
input_file_2020 = '/data/challenge_data/origin/chla/cmems_mod_glo_bgc_my_0.25deg_P1D-m_chl_180.00W-179.75E_80.00S-90.00N_0.51-53.85m_2020-01-01-2020-12-31.nc'
output_dir_2020 = '/data/challenge_data/processed/processed_chla/2020/' 

input_file_2019 = '/data/challenge_data/origin/chla/cmems_mod_glo_bgc_my_0.25deg_P1D-m_chl_180.00W-179.75E_80.00S-90.00N_0.51-53.85m_2019-01-01-2019-12-31.nc'
output_dir_2019 = '/data/challenge_data/processed/processed_chla/2019/'

# 处理2019年所有日期的数据
process_all_days(input_file_2019, output_dir_2019)
# 处理2020年所有日期的数据
process_all_days(input_file_2020, output_dir_2020)
