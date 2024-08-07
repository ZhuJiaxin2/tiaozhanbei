import os
import numpy as np
import torch
from netCDF4 import Dataset, num2date

def read_and_convert_variable(file_path, variable_name, block_size):
    with Dataset(file_path, 'r') as nc:
        var = nc.variables[variable_name]
        # 读取比例因子和偏移量属性，如果存在的话
        scale_factor = var.scale_factor if 'scale_factor' in var.ncattrs() else 1
        offset = var.add_offset if 'add_offset' in var.ncattrs() else 0
        data_shape = var.shape
        # 读取填充值
        fill_value = var._FillValue if '_FillValue' in var.ncattrs() else np.nan

        fill_value = fill_value * scale_factor + offset
        
        # 初始化一个空的数组来存储结果
        result = np.zeros(data_shape, dtype=np.float32)
        
        # 按块处理数据
        for i in range(0, data_shape[0], block_size):
            for j in range(0, data_shape[1], block_size):
                for k in range(0, data_shape[2], block_size):
                    i_end = min(i + block_size, data_shape[0])
                    j_end = min(j + block_size, data_shape[1])
                    k_end = min(k + block_size, data_shape[2])
                    
                    data_block = var[i:i_end, j:j_end, k:k_end]
                    # result[i:i_end, j:j_end, k:k_end] = (data_block-offset)/ scale_factor 
                    result[i:i_end, j:j_end, k:k_end] = data_block


        # 将填充值转换为 NaN
        result = np.where(result == fill_value, np.nan, result)
        
        return result

# def read_and_convert_variable(nc_file, var_name):
#     """
#     读取并转换 NetCDF 文件中的变量，考虑比例因子和偏移量。
    
#     参数：
#     nc_file (str): NetCDF 文件的路径。
#     var_name (str): 要读取的变量名。
    
#     返回：
#     data (numpy.ndarray): 转换后的数据。
#     """
#     dataset = Dataset(nc_file, 'r')
#     # print(dataset.variables.keys())
#     var = dataset.variables[var_name]
    
#     # 读取比例因子和偏移量属性，如果存在的话
#     scale_factor = var.scale_factor if 'scale_factor' in var.ncattrs() else 1
#     offset = var.add_offset if 'add_offset' in var.ncattrs() else 0
    
#     # 读取填充值
#     fill_value = var._FillValue if '_FillValue' in var.ncattrs() else np.nan

#     # 读取数据并进行转换
#     data = var[:] * scale_factor + offset
#     print(data._FillValue)
#     return
#     # fill_value = fill_value * scale_factor + offset
    
#     # 将填充值转换为 NaN
#     data = np.where(data == fill_value, np.nan, data)
    
#     dataset.close()
#     return data

# def process_all_days(file_path, output_dir):
def process_all_days(file_path, output_dir, block_size=100):
    """
    处理一个NetCDF文件中的所有天的数据提,取经纬度和叶绿素浓度数据并保存为Tensor文件。

    参数：
    file_path (str): 输入NetCDF文件的路径。
    output_dir (str): 输出Tensor文件的目录。
    """
    dataset = Dataset(file_path, 'r')
    
    # 读取变量
    latitude = dataset.variables['latitude'][:]
    longitude = dataset.variables['longitude'][:]
    time = dataset.variables['time']
    
    # 获取时间戳对应的日期
    dates = num2date(time[:], time.units)
    
    # 读取叶绿素浓度数据
    chl = read_and_convert_variable(file_path, 'analysed_sst', block_size)
    
    # 创建网格数据
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    
    for day_index, date in enumerate(dates):
        date_str = date.strftime("%Y-%m-%d")
        print(date_str)
        
        # 提取当前日期的叶绿素浓度数据
        chl_day = chl[day_index]
        
        # 将填充值替换为 NaN
        chl_day = np.where(chl_day == -32768, np.nan, chl_day)
        
        # 将数据转换为Tensor
        chl_tensor = torch.tensor(chl_day, dtype=torch.float32)
        
        # 输出文件路径
        output_file = os.path.join(output_dir, f"{date_str}.pt")
        
        # 保存Tensor数据
        tensor_data = chl_tensor

        torch.save(tensor_data, output_file)
        
    dataset.close()

# 示例调用
input_file_2020 = '/data/challenge_data/origin/sst/METOFFICE-GLO-SST-L4-REP-OBS-SST_analysed_sst_179.98W-179.98E_89.97S-89.97N_2020-01-01-2020-12-31.nc'
output_dir_2020 = '/data/challenge_data/processed/processed_sst/2020/' 

input_file_2019 = '/data/challenge_data/origin/sst/METOFFICE-GLO-SST-L4-REP-OBS-SST_analysed_sst_179.98W-179.98E_89.97S-89.97N_2019-01-01-2019-12-31.nc'
output_dir_2019 = '/data/challenge_data/processed/processed_sst/2019/'

# 处理2019年所有日期的数据
process_all_days(input_file_2019, output_dir_2019)
# 处理2020年所有日期的数据
process_all_days(input_file_2020, output_dir_2020)

